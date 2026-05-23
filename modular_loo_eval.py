import os
import sqlite3
import pandas as pd
import numpy as np
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import openai
import argparse

# 1. BOOTSTRAP ENVIRONMENT
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MODULAR-LOOE")

class ModularLOOE:
    def __init__(self, tickets_db, feedback_db, results_dir="test_results_definitive_looe"):
        self.tickets_db = tickets_db
        self.feedback_db = feedback_db
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize API Client
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
        
        self.model_name = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo-0613")
        
        # Load Tickets
        print("📁 Loading Tickets...")
        conn = sqlite3.connect(tickets_db)
        self.tickets_df = pd.read_sql_query("SELECT * FROM tickets", conn)
        conn.close()
        
        # Load Feedback
        print("📁 Loading Feedback Matrix...")
        conn = sqlite3.connect(feedback_db)
        self.feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()
        
        # Initialize Embeddings
        print("⚡ Initializing Vector Engine...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ticket_texts = [f"{str(r['title'])} {str(r['description'])}" for _, r in self.tickets_df.iterrows()]
        self.ticket_embeddings = self.embed_model.encode(self.ticket_texts, convert_to_tensor=True)
        
        # Map sequential_id to index for fast lookup (matching R-n style in feedback DB)
        self.id_to_idx = {str(sid): i for i, sid in enumerate(self.tickets_df['sequential_id'])}
        
        # Also map ticket_id (Ref) to index as fallback
        self.ref_to_idx = {str(tid): i for i, tid in enumerate(self.tickets_df['ticket_id'])}

    async def call_llm(self, prompt, temperature=0.0):
        """Self-contained LLM wrapper."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Call failed: {e}")
            return f"Error: {e}"

    async def judge_responses(self, query_text, ground_truth, baseline_resp, al_resp):
        """Use LLM to judge both responses against ground truth."""
        prompt = f"""
Evaluate two AI-generated resolutions for a technical support ticket against the ground truth resolution.

TICKET:
{query_text}

GROUND TRUTH RESOLUTION:
{ground_truth}

AI RESOLUTION A (Baseline):
{baseline_resp}

AI RESOLUTION B (Active Learning):
{al_resp}

Please score both AI resolutions from 0.0 to 1.0 based on how well they match the intent and accuracy of the ground truth.
Respond ONLY in JSON format:
{{"baseline_score": float, "al_score": float, "reasoning": "string"}}
"""
        raw_resp = await self.call_llm(prompt)
        try:
            # Clean up potential markdown code blocks
            clean_json = re.sub(r'```json\s*|\s*```', '', raw_resp).strip()
            # If still starts with ```, try harder
            if '```' in clean_json:
                 clean_json = clean_json.split('```')[1]
                 if clean_json.startswith('json'): clean_json = clean_json[4:]
            return json.loads(clean_json)
        except:
            logger.error(f"Failed to parse judge response: {raw_resp}")
            return {"baseline_score": 0, "al_score": 0, "reasoning": f"Parsing error: {raw_resp[:100]}"}

    def calculate_lifts(self, query_id, query_class):
        """
        Calculate AL lifts for all candidates based on feedback from other queries.
        LOO: Exclude feedback where query_id == query_id.
        """
        # Isolation: exclude current query's feedback
        train_mem = self.feedback_df[self.feedback_df['query_ticket_id'] != query_id].copy()
        
        # Optional: further filter by class if specified
        if query_class:
            train_mem = train_mem[train_mem['query_class'] == query_class]
            
        if train_mem.empty:
            return {}
            
        # Formula: Accumulated Evidence Tanh
        if 'similarity' in train_mem.columns:
            train_mem['ev'] = (train_mem['score'] - 0.5) * train_mem['similarity']
        else:
            train_mem['ev'] = (train_mem['score'] - 0.5)
            
        acc_ev = train_mem.groupby('feedback_ticket_id')['ev'].sum().reset_index()
        acc_ev['lift'] = np.tanh(acc_ev['ev'] / 5.0) * 0.15
        
        return dict(zip(acc_ev['feedback_ticket_id'], acc_ev['lift']))

    def get_retrieval(self, query_text, excluded_id, lift_map=None, top_k=5):
        """Perform retrieval with optional lifts."""
        query_embedding = self.embed_model.encode(query_text, convert_to_tensor=True)
        cos_sims = util.cos_sim(query_embedding, self.ticket_embeddings)[0]
        
        candidates = []
        for i, row in self.tickets_df.iterrows():
            tid = str(row['ticket_id'])
            sid = str(row['sequential_id'])
            if sid == excluded_id or tid == excluded_id:
                continue
                
            base_score = float(cos_sims[i])
            # Lift can be mapped via sequential_id or ticket_id
            lift = lift_map.get(sid, 0.0) if lift_map else 0.0
            if lift == 0 and lift_map:
                lift = lift_map.get(tid, 0.0)
                
            final_score = base_score + lift
            
            candidates.append({
                'ticket_id': tid,
                'sequential_id': sid,
                'title': row['title'],
                'description': row['description'],
                'first_reply': row['first_reply'],
                'base_score': base_score,
                'lift': lift,
                'final_score': final_score
            })
            
        # Sort and take top_k
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates[:top_k]

    def format_context(self, retrieved_items):
        context = ""
        for i, item in enumerate(retrieved_items):
            context += f"--- EXAMPLE {i+1} ---\n"
            context += f"ID: {item['ticket_id']} ({item['sequential_id']})\n"
            context += f"Title: {item['title']}\n"
            context += f"Description: {item['description']}\n"
            context += f"Resolution: {item['first_reply']}\n\n"
        return context

    async def run_loo_for_ticket(self, target_id):
        # lookup index
        if target_id not in self.id_to_idx:
            if target_id not in self.ref_to_idx:
                logger.warning(f"Target ID {target_id} not found in tickets.")
                return None
            idx = self.ref_to_idx[target_id]
        else:
            idx = self.id_to_idx[target_id]
            
        target_row = self.tickets_df.iloc[idx]
        query_text = f"{target_row['title']} {target_row['description']}"
        query_class = target_row['service_subcategory']
        
        print(f"🔍 Evaluating LOO for {target_id}...")
        
        # 1. Baseline (No Lift)
        baseline_retrieval = self.get_retrieval(query_text, target_id, lift_map=None)
        baseline_context = self.format_context(baseline_retrieval)
        baseline_prompt = f"Given the following previous tickets:\n{baseline_context}\nHow would you resolve this ticket?\nTitle: {target_row['title']}\nDescription: {target_row['description']}"
        baseline_response = await self.call_llm(baseline_prompt)
        
        # 2. AL (With Lift)
        lift_map = self.calculate_lifts(target_id, query_class)
        al_retrieval = self.get_retrieval(query_text, target_id, lift_map=lift_map)
        al_context = self.format_context(al_retrieval)
        al_prompt = f"Given the following previous tickets:\n{al_context}\nHow would you resolve this ticket?\nTitle: {target_row['title']}\nDescription: {target_row['description']}"
        al_response = await self.call_llm(al_prompt)
        
        # 3. Judge
        judgment = await self.judge_responses(query_text, target_row['first_reply'], baseline_response, al_response)
        
        # Result entry
        result = {
            'query_ticket_id': target_id,
            'sequential_id': target_row['sequential_id'],
            'query_title': target_row['title'],
            'query_description': target_row['description'],
            'ground_truth': target_row['first_reply'],
            'query_class': query_class,
            'baseline': {
                'retrieval': [{k: v for k, v in item.items() if k != 'description'} for item in baseline_retrieval],
                'response': baseline_response,
                'score': judgment.get('baseline_score', 0)
            },
            'al': {
                'retrieval': [{k: v for k, v in item.items() if k != 'description'} for item in al_retrieval],
                'response': al_response,
                'score': judgment.get('al_score', 0),
                'lift_applied_count': len([l for l in lift_map.values() if l != 0])
            },
            'judgment_reasoning': judgment.get('reasoning', ""),
            'metrics': {
                'delta': judgment.get('al_score', 0) - judgment.get('baseline_score', 0),
                'win': judgment.get('al_score', 0) > judgment.get('baseline_score', 0)
            }
        }
        
        return result

    async def run_full_eval(self, limit=None, filter_class=None, filter_team=None, context_desc=""):
        target_ids = self.feedback_df['query_ticket_id'].unique().tolist()
        
        if filter_class:
            filtered_df = self.tickets_df[self.tickets_df['service_subcategory'] == filter_class]
            target_ids = [tid for tid in target_ids if tid in filtered_df['sequential_id'].values or tid in filtered_df['ticket_id'].values]
        
        if filter_team:
            filtered_df = self.tickets_df[self.tickets_df['team'] == filter_team]
            target_ids = [tid for tid in target_ids if tid in filtered_df['sequential_id'].values or tid in filtered_df['ticket_id'].values]
            
        if limit:
            target_ids = target_ids[:limit]
            
        all_results = []
        for tid in target_ids:
            res = await self.run_loo_for_ticket(tid)
            if res:
                all_results.append(res)
                
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"looe_results_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'context_desc': context_desc,
            'config': {
                'limit': limit,
                'filter_class': filter_class,
                'filter_team': filter_team,
                'model': self.model_name
            },
            'stats': {
                'total_queries': len(all_results),
                'mean_baseline_score': np.mean([r['baseline']['score'] for r in all_results]) if all_results else 0,
                'mean_al_score': np.mean([r['al']['score'] for r in all_results]) if all_results else 0,
                'total_wins': len([r for r in all_results if r['metrics']['win']])
            },
            'results': all_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        print(f"✅ Evaluation complete. Results saved to {filename}")
        print(f"📊 Mean Baseline: {summary['stats']['mean_baseline_score']:.3f}, Mean AL: {summary['stats']['mean_al_score']:.3f}")
        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular LOO Evaluation Engine")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of tickets to evaluate")
    parser.add_argument("--class_filter", type=str, default=None, help="Filter by service subcategory")
    parser.add_argument("--team_filter", type=str, default=None, help="Filter by team")
    parser.add_argument("--context", type=str, default="Manual Run", help="Context description for this evaluation")
    
    args = parser.parse_args()
    
    engine = ModularLOOE(
        tickets_db="tickets.db", 
        feedback_db="comprehensive_feedback_250x250.db"
    )
    
    asyncio.run(engine.run_full_eval(
        limit=args.limit, 
        filter_class=args.class_filter,
        filter_team=args.team_filter,
        context_desc=args.context
    ))

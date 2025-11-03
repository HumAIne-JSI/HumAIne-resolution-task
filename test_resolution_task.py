import pandas as pd
from sentence_transformers import SentenceTransformer, util
from resolution_task import process_new_ticket
import re

# Define the path to the original knowledge base
original_knowledge_base_path = "tickets_large_first_reply_label.csv"

# Load the original dataset
df = pd.read_csv(original_knowledge_base_path)

# Define a list of test tickets (examples from the dataset)
test_tickets_new = [
    {
        "title": row["Title_anon"],
        "description": row["Description_anon"],
    }
    for _, row in df.sample(80, random_state=42).iterrows()  # Randomly select 10 examples
]

# Remove the test examples from the dataset copy
test_indices = [df.index[df["Title_anon"] == ticket["title"]].tolist()[0] for ticket in test_tickets_new]
df_copy = df.drop(test_indices).reset_index(drop=True)

# Save the modified dataset copy
knowledge_base_path = "tickets_large_first_reply_label_copy.csv"
df_copy.to_csv(knowledge_base_path, index=False)

# # Initialize Sentence-BERT model for evaluation
# # sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Run the test cases
# df = df.dropna(subset=['Public_log_anon'])

# # Extract proper first replies using the same logic as the fixed function
# def extract_first_reply_only(text):
#     """Extract ONLY the first reply from chat log, not entire history"""
#     if pd.isna(text):
#         return None
    
#     text_str = str(text)
    
#     # More aggressive pattern to find timestamp separators and user info
#     timestamp_pattern = r'\*{10,}\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'
    
#     # Also look for patterns like ": servicedesk (0) ************" or ": Name (ID) ************"
#     user_pattern = r':\s*[^(]*\([^)]*\)\s*\*{10,}'
    
#     # Split by timestamp pattern first
#     parts = re.split(timestamp_pattern, text_str)
    
#     if len(parts) >= 2:
#         first_reply = parts[1].strip()
        
#         # Remove user info pattern at the beginning
#         first_reply = re.sub(user_pattern, '', first_reply, flags=re.IGNORECASE).strip()
        
#         # Find the next timestamp to cut off subsequent replies
#         next_timestamp_match = re.search(timestamp_pattern, first_reply)
#         if next_timestamp_match:
#             first_reply = first_reply[:next_timestamp_match.start()].strip()
        
#         # Find next user pattern to cut off subsequent replies
#         next_user_match = re.search(user_pattern, first_reply)
#         if next_user_match:
#             first_reply = first_reply[:next_user_match.start()].strip()
        
#         # Clean up common artifacts
#         first_reply = re.sub(r'^-+\s*', '', first_reply)  # Remove leading dashes
#         first_reply = re.sub(r'\s*-+$', '', first_reply)  # Remove trailing dashes
#         first_reply = re.sub(r'^\s*Dear\s+[^,]+,?\s*', '', first_reply, flags=re.IGNORECASE)  # Remove "Dear Name," at start
        
#         # Remove lines that start with specific patterns
#         lines = first_reply.split('\n')
#         cleaned_lines = []
#         for line in lines:
#             line = line.strip()
#             # Skip lines that are just separators or user info
#             if not re.match(r'^-{5,}$', line) and not re.match(r'^\s*:\s*[^(]*\([^)]*\)', line):
#                 cleaned_lines.append(line)
        
#         first_reply = '\n'.join(cleaned_lines).strip()
        
#         # Return only if it's substantial (not just whitespace or artifacts)
#         return first_reply if len(first_reply) > 50 else None
    
#     return text_str[:500] if len(text_str) > 50 else None

# # Apply the proper extraction
# df['first_reply'] = df['Public_log_anon'].apply(extract_first_reply_only)
# df = df.dropna(subset=['first_reply'])

# # Define a list of test tickets (examples from the dataset)
# test_tickets = [
#     {
#         "title": row["Title_anon"],
#         "description": row["Description_anon"],
#         "expected_first_reply": row["first_reply"]  # Now this is ONLY the first reply
#     }
#     for _, row in df.sample(5, random_state=42).iterrows()  # Randomly select 15 examples
# ]

# # Remove the test examples from the dataset copy
# test_indices = [df.index[df["Title_anon"] == ticket["title"]].tolist()[0] for ticket in test_tickets]
# df_copy = df.drop(test_indices).reset_index(drop=True)

# # Save the modified dataset copy
# knowledge_base_path = "tickets_large_first_reply_label_copy.csv"
# df_copy.to_csv(knowledge_base_path, index=False)

# # Initialize Sentence-BERT model for evaluation
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Run the test cases
# def test_rag_system():
#     results = []
#     cosine_similarities = []

#     for i, ticket in enumerate(test_tickets, 1):
#         print(f"\n{'='*60}")
#         print(f"Processing Ticket {i}: {ticket['title']}")
#         print(f"{'='*60}")
#         result = process_new_ticket(ticket["title"], ticket["description"], knowledge_base_path)
#         print("Predicted Class:", result["classification"])
#         print("Predicted Team:", result["predicted_team"])
#         print("Generated Response:", result["response"])
#         print("Expected First Reply:", ticket["expected_first_reply"][:200] + "...")  # Show only first 200 chars

#         # Compute cosine similarity between expected and generated first replies
#         expected_embedding = sentence_model.encode(ticket["expected_first_reply"], convert_to_tensor=True)
#         generated_embedding = sentence_model.encode(result["response"], convert_to_tensor=True)
#         cosine_similarity = util.cos_sim(expected_embedding, generated_embedding).item()
#         cosine_similarities.append(cosine_similarity)
#         print(f"Cosine Similarity: {cosine_similarity:.4f}")

#         results.append({
#             "title": ticket["title"],
#             "description": ticket["description"],
#             "expected_first_reply": ticket["expected_first_reply"],
#             "generated_response": result["response"],
#             "cosine_similarity": cosine_similarity
#         })

#     # Calculate average cosine similarity
#     avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
#     print(f"\nAverage Cosine Similarity: {avg_cosine_similarity:.4f}")

#     return results

# if __name__ == "__main__":
#     test_results = test_rag_system()
#     # Save the results to a file for further analysis
#     pd.DataFrame(test_results).to_json("test_results.json", orient="records", indent=4)




import pandas as pd
from sentence_transformers import SentenceTransformer, util
from resolution_task import process_new_ticket
import re
import os
import pickle
import numpy as np
import faiss
import json
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Define the path to the original knowledge base
original_knowledge_base_path = "tickets_large_first_reply_label.csv"

# Cache directory for embeddings
CACHE_DIR = Path("./embeddings_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_filename(knowledge_base_path):
    """Generate cache filename based on knowledge base file"""
    kb_file = Path(knowledge_base_path)
    return CACHE_DIR / f"rag_system_{kb_file.stem}.pkl"

def load_cached_rag_system(knowledge_base_path):
    """Load cached RAG system if available"""
    cache_file = get_cache_filename(knowledge_base_path)
    
    if cache_file.exists():
        try:
            print(f"📂 Loading cached RAG system from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print("✅ RAG system loaded from cache")
            return cached_data
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
            return None
    return None

def save_rag_system_to_cache(rag_system, knowledge_base_path):
    """Save RAG system to cache"""
    cache_file = get_cache_filename(knowledge_base_path)
    
    try:
        print(f"💾 Saving RAG system to cache: {cache_file}")
        
        # Create a serializable version of the RAG system
        cache_data = {
            'knowledge_base': rag_system.knowledge_base,
            'embeddings': rag_system.embeddings,
            'title_embeddings': rag_system.title_embeddings,
            'description_embeddings': rag_system.description_embeddings,
            'category_index': rag_system.category_index,
            'model_name': rag_system.sentence_model._modules['0'].auto_model.name_or_path
        }
        
        # Save FAISS index separately
        faiss_file = cache_file.with_suffix('.faiss')
        faiss.write_index(rag_system.index, str(faiss_file))
        cache_data['faiss_file'] = str(faiss_file)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print("✅ RAG system saved to cache")
        
    except Exception as e:
        print(f"❌ Failed to save cache: {e}")

def create_rag_system_from_cache(cached_data, sentence_model_name="all-MiniLM-L6-v2"):
    """Recreate RAG system from cached data"""
    from resolution_task import RAGSystem
    from sentence_transformers import SentenceTransformer
    
    # Create RAG system instance
    rag_system = RAGSystem(cached_data['knowledge_base'], sentence_model_name)
    
    # Restore cached data
    rag_system.embeddings = cached_data['embeddings']
    rag_system.title_embeddings = cached_data['title_embeddings']
    rag_system.description_embeddings = cached_data['description_embeddings']
    rag_system.category_index = cached_data['category_index']
    
    # Load FAISS index
    rag_system.index = faiss.read_index(cached_data['faiss_file'])
    
    return rag_system

# Global variable to store the RAG system
_global_rag_system = None
_global_knowledge_base_path = None

def get_or_create_rag_system(knowledge_base_path):
    """Get cached RAG system or create new one"""
    global _global_rag_system, _global_knowledge_base_path
    
    # If we already have the right RAG system loaded, return it
    if _global_rag_system is not None and _global_knowledge_base_path == knowledge_base_path:
        return _global_rag_system
    
    # Try to load from cache first
    cached_data = load_cached_rag_system(knowledge_base_path)
    
    if cached_data is not None:
        _global_rag_system = create_rag_system_from_cache(cached_data)
        _global_knowledge_base_path = knowledge_base_path
        return _global_rag_system
    
    # If no cache, create new RAG system
    print("🔄 Creating new RAG system (no cache found)")
    from resolution_task import load_knowledge_base, RAGSystem
    
    knowledge_base = load_knowledge_base(knowledge_base_path)
    _global_rag_system = RAGSystem(knowledge_base)
    _global_rag_system.build_index()
    
    # Save to cache for future use
    save_rag_system_to_cache(_global_rag_system, knowledge_base_path)
    
    _global_knowledge_base_path = knowledge_base_path
    return _global_rag_system

# Load the original dataset
print("📂 Loading original dataset...")
df_original = pd.read_csv(original_knowledge_base_path)
df_original = df_original.dropna(subset=['Public_log_anon'])

def extract_first_reply_only(text):
    """Extract ONLY the first reply from chat log, not entire history"""
    if pd.isna(text):
        return None
    
    text_str = str(text)
    
    # More aggressive pattern to find timestamp separators and user info
    timestamp_pattern = r'\*{10,}\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'
    
    # Also look for patterns like ": servicedesk (0) ************" or ": Name (ID) ************"
    user_pattern = r':\s*[^(]*\([^)]*\)\s*\*{10,}'
    
    # Split by timestamp pattern first
    parts = re.split(timestamp_pattern, text_str)
    
    if len(parts) >= 2:
        first_reply = parts[1].strip()
        
        # Remove user info pattern at the beginning
        first_reply = re.sub(user_pattern, '', first_reply, flags=re.IGNORECASE).strip()
        
        # Find the next timestamp to cut off subsequent replies
        next_timestamp_match = re.search(timestamp_pattern, first_reply)
        if next_timestamp_match:
            first_reply = first_reply[:next_timestamp_match.start()].strip()
        
        # Find next user pattern to cut off subsequent replies
        next_user_match = re.search(user_pattern, first_reply)
        if next_user_match:
            first_reply = first_reply[:next_user_match.start()].strip()
        
        # Clean up common artifacts
        first_reply = re.sub(r'^-+\s*', '', first_reply)  # Remove leading dashes
        first_reply = re.sub(r'\s*-+$', '', first_reply)  # Remove trailing dashes
        first_reply = re.sub(r'^\s*Dear\s+[^,]+,?\s*', '', first_reply, flags=re.IGNORECASE)  # Remove "Dear Name," at start
        
        # Remove lines that start with specific patterns
        lines = first_reply.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are just separators or user info
            if not re.match(r'^-{5,}$', line) and not re.match(r'^\s*:\s*[^(]*\([^)]*\)', line):
                cleaned_lines.append(line)
        
        first_reply = '\n'.join(cleaned_lines).strip()
        
        # Return only if it's substantial (not just whitespace or artifacts)
        return first_reply if len(first_reply) > 50 else None
    
    return text_str[:500] if len(text_str) > 50 else None

# Extract first replies from original dataset
print("🔄 Extracting first replies from original dataset...")
df_original['first_reply'] = df_original['Public_log_anon'].apply(extract_first_reply_only)
df_original = df_original.dropna(subset=['first_reply'])
print(f"✅ Dataset processed: {len(df_original)} tickets with valid first replies")

hardcoded_test_tickets = test_tickets_new #[
#     {
#         "title": "Admin Rights to setup the new PC",
#         "description": "I need a admin rights for install a software for work."
#     },
#     {
#         "title": "Need administrator rights for Java Academy",
#         "description": "I need Administrator rights for the Java Academy."
#     },
#     {
#         "title": "I need administrator rights on my computer",
#         "description": "I need administrator rights on my computer to install the new version of Ivanti"
#     },
#     {
#         "title": "Badge to GFT Italy - urgente",
#         "description": "Fagandini Ruiz RenzoXYZ123TorinoGFT Italia14455274\nRuggeri GiuseppeABC456MilanoGFT Italia14557493\nRiva Maria CristinaDEF789MilanoGFT Italia14352415\nNegretti AldoGHI012MilanoGFT Italia14595748\nDi Vita Gaia AntonellaJKL345TorinoGFT Italia14586489\nZuardi SalvatoreMNO678FirenzeGFT Italia14444587\nErcole LorenzoPQR901TorinoGFT Italia14257369"
#     },
#     {
#         "title": "Administrator rights and software installation",
#         "description": "I need Administrator rights and software installation on my computer to install Microsoft Office 365, Visual Studio Code, Chrome, Firefox, Notepad++, WinSCP, Putty, Autodesk Inventor 2023, Teams client for audio conferences I'have request on 3 different times for a response but still have not received one. if you could please expedite this request."
#     },
#     {
#         "title": "shared mailbox is slow",
#         "description": "shared mailbox pseudomailbox1.supporto@example.com is slow in loading and searching items"
#     },
#     {
#         "title": "Invoices missing INV136364 INV134830",
#         "description": "Hi All, the below invoices are missing during send to ERP\n\nPlease can you verify and post to Sap?\n\nThanks in advance\n\nBest regards\n\nAbigail"
#     },
#     {
#         "title": "need admin permission",
#         "description": "i need admin permission for new project"
#     },
#     {
#         "title": "Request for Administrator Rights to Set Java Environment Variable",
#         "description": "Hello. I need to add a global environment variable for Java so I can execute my JAR files using the command prompt on my PC. Could I please have administrator rights to create this setting? Thank you."
#     },
#     {
#         "title": "I renew the request for obtain administrator rights",
#         "description": "I request the renewal of the admin rights on the two Laptop (old and new in preparation) indicated in the ticket."
#     },
#     {
#         "title": "I need a new network access for UnipolSai",
#         "description": "Hi,\nI need to add some IP access for UnipolSai for everyone in GFT VPN.\n\nTool: Gitea\nIP: 192.168.1.1\nPORT: 3000\nHOST: node1.example.com\nENV: Global\n\nTool: Jenkins\nIP: 192.168.1.1\nPORT: 49000\nHOST: node1.example.com\nENV: Global\n\nTool: Sonarqube\nIP: 192.168.1.1\nPORT: 9092\nHOST: node1.example.com\nENV: Global\n\nTool: Nexus\nIP: 192.168.1.1\nPORT: 8081\nHOST: node1.example.com\nENV: Global\n\nThank you,\nJohn"
#     },
#     {
#         "title": "Gft Italia_new_entry_20240708_992412_ Bianchi",
#         "description": "Request forON - Boarding\n\nemployee datainternal / externalInternal employee\ndate of entry8 Jul 2024\nSurnameBianchi\nNameLuca\nprivate email addresslucabianchi99@example.com\nex external consultantno\nold GFT Code\nAccount information & permissionGFT code992412\nVPN ACCESS (requires policy signature)YES\nadditional permission\nCompany informationCompanyABC ITALIA SRL\nRoleRecruiter\nLevelL1\nClient unit\nManagerGiulia Moresi\nadditional info\nGFT officeLocationMilan\n1°ST Day LocationMilan\nRoom/Workplace/Address\nadditional info\nPersonal mobile number339 8756432\nAssetsPCs (include mouse + backpack)Laptop\ncompany mobile phoneNO\ntabletNO\n4G/UMTS USB keyNO\nother hardware / non standard softwareNO"
#     },
#     {
#         "title": "Old static config. on Ita Edge Switches removal",
#         "description": "Hello! Could you please remove these OLD static configurations in ITA Edge Switches?\n\nswxplr1-1 port 7 (Old Mexal server removed and disposed)\nswxplr1-1 port 8 (Old Mexal server removed and disposed)\nswxplr1-1 port 11 (Old Access control removed and disposed)\n\nswxplr1-2 port 8 (Old Access control removed and disposed)\n\nswxlan1-1 port 45 (Old Access control removed and disposed)\n\nswxdre1-2 port 43 (Old Access control removed and disposed)\n\nswklm1-1 port 42 (Old Access control removed and disposed)\n\nswxyza1-3 port 36 (Old Access control removed and disposed)\n\nswqwe1-2 port 48 (Old Access control removed and disposed)\n\nswxyzb1-3 port 22 (Old PRT-MLN-F4-OPENSPACE removed and disposed)\n\nswxyzb2-1 port 4 (Old PRT-MLN-F4-Laudisa removed and disposed)\n\nI remain available.\n\nBr\n\nJohn"
#     },
#     {
#         "title": "Gft Italia_new_entry_20240708_643298_Rossi",
#         "description": "Request forON - Boarding\n\nemployee datainternal / externalInternal employee\ndate of entry8 Jul 2024\nSurnameRossi\nNameAlessandro\nprivate email addressalessandro.rossi@example.com\nex external consultant\nold GFT Code\nAccount information & permissionGFT code643298\nmember of the DL spcdipendentiYES\nVPN ACCESS (requires policy signature)YES\nadditional permission\nCompany informationCompanyRND ITALIA SRL\nRoleDATA\nLevelL6\nClient unitDATA PRACTICE\nManagerMARCO DE LUCA\nadditional info\nGFT officeLocationMilan\n1°ST Day LocationMilan\nRoom/Workplace/Address\nadditional info\nPersonal mobile number+39 3285732004\nAssetsPCs (include mouse + backpack)Laptop\ncompany mobile phoneNO\ntabletNO\n4G/UMTS USB keyNO\nother hardware / non standard softwareNO"
#     },
#     {
#         "title": "Ivanti Software Installation",
#         "description": "I need the administrator permission to download the ivanti software necessary to connect to the client's VPN"
#     },
#     {
#         "title": "Restore file of an ex-employee",
#         "description": "Hi,\\n\\nI need the following file:\\n\\nshort.link/placeholder\\n\\nIs it possible to receive it?\\n\\nThanks"
#     },
#     {
#         "title": "I need to stop automated updates to node js",
#         "description": "I need to stop automated updates to node js for my computer (a1b2) and Luca Valente (c3d4)"
#     },
#     {
#         "title": "Enable user to access client network (Albaleasing)",
#         "description": "Hi, we need to give access to Albaleasing client network to the new colleague Marco Rossi (marco.rossi@xyz.com).\\n\\nWe use a L2L connection via XYZ VPN."
#     },
#     {
#         "title": "I need administrator rights on my computer",
#         "description": "I need permissions to install Ivanti /VPN/Oracle Db software for the project"
#     },
#     {
#         "title": "Gft italia_JOHNSON ALICE_B45B",
#         "description": "HI,\\n\\nI would need to cancel Johnson Alice's absence of 4 hours on 13th June and to insert an absence of 2 hours in the same day\\n\\nI attach the screen of the canceled request and the new request on Success Factors.\\n\\nMany Thanks,\\n\\nLina"
#     },
#     {
#         "title": "Setup backend environment for new pc",
#         "description": "I need to install a list of development-related software for my new PC"
#     },
#     {
#         "title": "GFT VPN not present on my PC",
#         "description": "The GFT VPN is not present in my PC (Again...)"
#     },
#     {
#         "title": "Gft Italia__new_dismissal__20240604_ceoe_Williams [Ticket to manage device retirements]",
#         "description": "Request forOFF - Boarding\\n\\nemployee datainternal / externalInternal employee\\ndate of exit4 Jun 2024\\nSurnameWilliams\\nNameHenry Grant\\nprivate email address\\nex externo\\nadditional info\\nAccount information & permissionGFT code193764\\n\\nVPN ACCESS (requires policy signature)\\nadditional permission\\nCompany informationCompanyGFT ITALIA SRL\\nRole\\nLevel\\nClient unit\\nManagerRoss Michael\\nadditional info\\nGFT officeLocationFlorence\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info\\nAssetsPCs (include mouse + backpack)\\ncompany mobile phone\\ntablet\\n4G/UMTS USB key\\nother hardware / non standard software"
#     },
#     {
#         "title": "I need support about local software",
#         "description": "I need support about local software"
#     },
#     {
#         "title": "Need Admin Rights",
#         "description": "Hi\\n\\nI need administration rights on my PC in order to change env variables. Thank you"
#     },
#     {
#         "title": "Update working tools",
#         "description": "I need admin rights for updating my working tools, vpn, environment, software etc"
#     },
#     {
#         "title": "gft_italia_new_entry_20240610_381492_FRANCHI",
#         "description": "Request for ON - Boarding\\n\\nemployee data internal / external Consultant\\ndate of entry 10 Jun 2024\\nSurname FRANCHI\\nName SOFIA\\nprivate email address sofia.franchi@randommail.com\\nadditional info\\nAccount information & permission GFT code\\n\\nVPN ACCESS (requires policy signature) YES\\nadditional permission\\nCompany information Company GFT ITALIA SRL\\nRole\\nSAP code 381492\\nLevel\\nClient unit BPER\\nManager RUSSO\\nadditional info\\nGFT office Location Milan\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info"
#     },
#     {
#         "title": "Gft Italia_new_entry_20240715_123456_ Rossi",
#         "description": "Request forON - Boarding\\n\\nemployee datainternal / externalInternal employee\\ndate of entry\\nSurnameRossi\\nNameMario\\nprivate email addressmariorossi@example.com\\nex external consultantno\\nold GFT Code\\nAccount information & permissionGFT code123456\\nVPN ACCESS (requires policy signature)YES\\nadditional permission\\nCompany informationCompanyABC ITALIA SRL\\nRoleBE\\nLevelL2\\nClient unit\\nManagerGianni Bianchi\\nadditional info\\nGFT officeLocationMilan\\n1°ST Day LocationMilan\\nRoom/Workplace/Address\\nadditional info\\nPersonal mobile number3331234567\\nAssetsPCs (include mouse + backpack)Laptop\\ncompany mobile phoneNO\\ntabletNO\\n4G/UMTS USB keyNO\\nother hardware / non standard softwareNO"
#     },
#     {
#         "title": "I need administrator permissions to download JAVA jdk",
#         "description": "I need administrator permissions to download JAVA jdk"
#     },
#     {
#         "title": "I need permanent admin rights",
#         "description": "Hello,\\nI need admin rights to install and update docker.\\nThank in advance,\\nLeonardo Rossi."
#     },
#     {
#         "title": "I need access MYCO Platform",
#         "description": "Hello,\\nI need to be activate to Myco Platform"
#     },
#     {
#         "title": "Setup PC",
#         "description": "I need admin rights to install wsl and other software. My LM is Alex Morandi. I need to setup my PC for the project"
#     },
#     {
#         "title": "New UnipolSai Endpoint EURO CLIENTI",
#         "description": "Real IP of the Customer server: 192.168.1.1\\nHostname: euro.retailcompany.com\\nServer Description: ENDPOINT EURO CLIENTI\\nPorts to be opened: HTTPS(443)\\nPorts to be opened: HTTP(80)\\nUnipol manager references VERDI/ROSSI/BIANCHI\\nUnipol Project 2023 - UnipolSai - Leonardo - Evolutive R8 (KE-038971)"
#     },
#     {
#         "title": "gft_italia_new_dismiss_20240630_A9K7X2 [Ticket to manage device retirements]",
#         "description": "Request forOFF - Boarding\\n\\nemployee datainternal / externalConsultant\\ndate of exit30 Jun 2024\\nSurnameROSSI\\nNameLUCA\\nprivate email address\\nadditional info\\nAccount information & permissionGFT codeA9K7X2\\n\\nVPN ACCESS (requires policy signature)YES\\nadditional permission\\nCompany informationCompanyXYZ ITALIA SRL\\nRole\\nSAP codeA9K7X2\\nLevel\\nClient unitCCB\\nManagerBERTI\\nadditional info\\nGFT officeLocationMilan\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info"
#     },
#     {
#         "title": "I need a confluence license",
#         "description": "I need a confluence license"
#     },
#     {
#         "title": "VM creation",
#         "description": "To deliver a project in CREDEM we need the installation of a Windows Virtual Machine"
#     },
#     {
#         "title": "Request for administrative rights",
#         "description": "I need administrative rights to install Maven, which I will need to use for a course that was assigned to me by my manager"
#     },
#     {
#         "title": "i need Admin rights in order to install several tools needed for Zenit and switch between jdk versions",
#         "description": "i need Admin rights in order to install the following tools needed for Zenit:\\n\\nDBeaver\\n\\nSoapUI\\n\\nInformatica PowerCenter\\n\\nPostman\\n\\nMoreover, i have been told that in the projects i will need to switch between jdk versions quite often."
#     },
#     {
#         "title": "VPN connected but unable to access Unipol services",
#         "description": "Good morning, I require assistance on the following issue:\\nI am connected to the XZY VPN but I cannot access the Unipol services (Database, Power Center Repository).\\n\\nCan you check if it's a VPN problem?\\n\\nThanks in advance"
#     },
#     {
#         "title": "Gft Italia__new_dismissal__20240718_a71f_Coi [Ticket to manage Admin Location Infrastructure Management Italy task]",
#         "description": "Request forOFF - Boarding\\n\\nemployee datainternal / externalInternal employee\\ndate of exit18 Jul 2024\\nSurnameAnselmi\\nNamePio\\nprivate email address\\nex externo\\nadditional info\\nAccount information & permissionGFT code123456\\n\\nVPN ACCESS (requires policy signature)\\nadditional permission\\nCompany informationCompanyABC ITALIA SRL\\nRole\\nLevel\\nClient unit\\nManagerBianchi Fabrizio\\nadditional info\\nGFT officeLocationMilan\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info\\nAssetsPCs (include mouse + backpack)\\ncompany mobile phone\\ntablet\\n4G/UMTS USB key\\nother hardware / non standard software"
#     },
#     {
#         "title": "Azure repo enablement for new e-lcbe team member",
#         "description": "Hi, our colleague Mario Rossi (matr: e-lcbe) has joined our team\\nand currently does not see the Azure Repo Service even though it appears to be added to project KE-039931-001. Can you please enable it.\\nThank you.\\nGiovanni."
#     },
#     {
#         "title": "gft_italia_new_dismiss_20240705_441729_E-JKOF [Ticket to manage virtual devices retirement]",
#         "description": "Request forOFF - Boarding\\n\\nemployee datainternal / externalConsultant\\ndate of exit5 Jul 2024\\nSurnameVELASCO\\nNameLUCIA\\nprivate email address\\nadditional info\\nAccount information & permissionGFT codeE-JKOF\\n\\nVPN ACCESS (requires policy signature)YES\\nadditional permission\\nCompany informationCompanyNUM ITALIA SRL\\nRole\\nSAP code441729\\nLevel\\nClient unitISP\\nManagerGUALTIERI\\nadditional info\\nGFT officeLocationTurin\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info"
#     },
#     {
#         "title": "Offboard employee Dario Fiore on 2024-07-16 [Ticket to manage virtual devices retirement]",
#         "description": "Initials: DAFO\\nFirst Name: Dario\\nLast Name: Fiore\\nSF employee ID: 419876\\nLeaving date: 2024-07-16\\nCountry: ITA\\nLegal Entity: IT02\\nLocation: Firenze\\nEmployee Level: L1 (Entry)\\nLine manager: Lucia Verdi (Milano)"
#     },
#     {
#         "title": "I need a license to access Jira",
#         "description": "I would need a license to access the dashboard of my team on Jira and manage my tasks for the project Trade Finance"
#     },
#     {
#         "title": "gft_italia_new_dismiss_814936_X-ASEQ [Ticket to manage Admin Location Infrastructure Management Italy task]",
#         "description": "Request forOFF - Boarding\\n\\nemployee datainternal / externalConsultant\\ndate of exit30 Jun 2024\\nSurnameFERRARO\\nNameLUIGI\\nprivate email address\\nadditional info\\nAccount information & permissionGFT codeX-ASEQ\\n\\nVPN ACCESS (requires policy signature)YES\\nadditional permission\\nCompany informationCompanyTECH ITALIA SRL\\nRole\\nSAP code873920\\nLevel\\nClient unitMICRO FOCUS\\nManagerGIORDANO\\nadditional info\\nGFT officeLocationMilan\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info"
#     },
#     {
#         "title": "Access to BPER network via GFT VPN",
#         "description": "Hello,\\n\\nI would need to enable my colleague Casey Hart (UYT6) to access to BPER Network via GFT VPN like other colleagues (i.e. Blake Spencer)\\n\\nMany thanks in advance.\\n\\nKind regards,\\n\\nMartha"
#     },
#     {
#         "title": "Access to BPER network via GFT VPN",
#         "description": "\\nI need to access to BPER network via GFT VPN like other colleagues (e.g. Luca Giannini).\\n"
#     },
#     {
#         "title": "GFT Italy: element activation problem",
#         "description": "Hi,\\nwe can't activate the element KE047825-010: the opportunity is set to 100% in status Closed Won but the probability in Kimble remains at 60% instead of 100%.\\nThank you\\nbest regards\\nMarco"
#     },
#     {
#         "title": "gft_italia_new_entry_20240604_546788_LOMBARDO [Ticket to manage device assignments]",
#         "description": "Request forON - Boarding\\n\\nemployee data internal / externalConsultant\\ndate of entry4 Jun 2024\\nSurnameLOMBARDO\\nNameMARCO\\nprivate email addressm.lombardo@corporateemail.com\\nadditional info\\nAccount information & permissionGFT code\\n\\nVPN ACCESS (requires policy signature)YES\\nadditional permission\\nCompany informationCompanyGFT ITALIA SRL\\nRole\\nSAP code546788\\nLevel\\nClient unitISP\\nManagerROSSI\\nadditional info\\nGFT officeLocationMilan\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info"
#     },
#     {
#         "title": "My PC often losts connection",
#         "description": "Sono connessa alla fibra, ma spesso il mio PC si disconnette.\\n\\nQuesto mi rende difficile lavorare"
#     },
#     {
#         "title": "I need a Jira license for LUCIA BERTOLLI",
#         "description": "I need a Jira license for LUCIA BERTOLLI"
#     },
#     {
#         "title": "Memory expansion",
#         "description": "Memory expansion to 512 GB for the 5540 id: ITPC024511 to replace the current device"
#     },
#     {
#         "title": "gft_italia_new_dismiss_20240425_E-XRTY_621345 [Ticket to manage Admin Location Infrastructure Management Italy task]",
#         "description": "Request forOFF - Boarding\\n\\nemployee datainternal / externalConsultant\\ndate of exit25 Apr 2024\\nSurnameBEXRE\\nNameKENNETH\\nprivate email address\\nadditional info\\nAccount information & permissionGFT codeE-XRTY\\n\\nVPN ACCESS (requires policy signature)YES\\nadditional permission\\nCompany informationCompanyGFX ITALIA SRL\\nRole\\nSAP code621345\\nLevel\\nClient unitISP\\nManagerREOLENS\\nadditional info\\nGFT officeLocationMilan\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info"
#     },
#     {
#         "title": "Substitution old PC",
#         "description": "Il tuo PC risulta obsoleto e fuori garanzia. Ti cambieremo il vecchio PC Service TAG 8GFD234 con uno nuovo.\\n\\nTi abbiamo assegnato il nuovo PC con Service TAG 7KDFS74\\n\\nProvvederemo a spedirti il nuovo PC nella filiale di Milano. Puoi darci conferma della filiale?\\n\\nA breve riceverai due mail:\\n\\n1) Una relativa alla nuova assegnazione\\n\\n2) Una relativa alla restituzione del tuo attuale device (da non considerare in caso hai esercitato il diritto di prelazione)\\n\\nQuesto ticket verrà aggiornato quando il PC sarà disponibile per il ritiro in filiale.\\n\\nTi ricordiamo che hai la possibilità di riscattare il tuo vecchio PC. Per esercitare la prelazione è sufficiente confermarlo nel ticket, successivamente verrai contattato per eseguire una formattazione da remoto. Se ricevi una mail dove ti viene chiesto di restituire il vecchio, non preoccuparti puoi ignorarla.\\n\\nIl prezzo è di 60 € per PC e 160 € per MacOS.\\n\\nNOTE: Ricorda che la prelazione è valida se:\\n\\nEntro 7 giorni dalla data di creazione del presente ticket eserciti il tuo diritto di prelazione.\\n\\nIl vecchio PC viene formattato/riconsegnato all'ufficio IT entro e non oltre 30 giorni dalla data di ritiro del nuovo device.\\n\\nLink utili:\\n\\n-\\nPolicy di assegnazione dei dispositivi aziendali https://short.link/policy-aziendali\\n\\n-\\nGuide IT https://short.link/it-guides\\n\\nL'IT provvederà alla cancellazione sicura dei tuoi dati ed al ripristino come condizioni di fabbrica del tuo vecchio pc.\\n\\n-------------------------------------------------\\n\\nYour PC is obsolete and out of warranty. We will change your old device with Service TAG 8GFD234 to a new one.\\n\\nWe have assigned you the new device with Service TAG 7KDFS74\\n\\nWe will send you the new PC to the Milan office. Can you give us the confirmation of the branch?\\n\\nYou will receive two emails:\\n\\n1) Regarding the new PC assigned to you\\n\\n2) Regarding the return of the old one (Please ignore if you already exercise the redeem old PC)\\n\\nThis ticket will be updated when the device can be picked up in the office.\\n\\nWe remind you that you can redeem your old PC. To exercise the pre-emption it's sufficient to confirm it in the ticket, after that we will contact you for resetting your old device remotely. If you'll receive an email asking you to return the old one, please ignore it.\\n\\nThe price is 60 € for PC and 160 € for MacOS.\\n\\nNOTES: Remember that the pre-emption is valid if:\\n\\nYou exercise the redemption of the old device within 7 days from the date of creation of this ticket.\\n\\nThe old PC will be reset/returned to the IT office within 30 days from the date you picked up the new device.\\n\\nUseful links:\\n\\n-\\nDevice Policy https://short.link/device-policy\\n\\n-\\nIT Guides: https://short.link/it-guides\\n\\nThe IT will provide for the secure deletion of your data and the restoration of your old PC to factory conditions."
#     },
#     {
#         "title": "OneDrive doesn't sync",
#         "description": "They keep turning the little arrows but there is always the words \\“Signing in.”\\n I have already tried closing the app and restarting the mac."
#     },
#     {
#         "title": "Lost access to 192.168.1.1",
#         "description": "Since the beginning of August we can no longer access the SFTP repository of PentaBank through the IP 192.168.1.1\\nThe IP in question is an internal FTI address that masks the address of the customer with which a VPN S2S tunnel is in place.\\nCan you verify which IP address corresponds on the customer side and the reason for the loss of access?\\nAttached is a connection screen of the client used with the configurations used and previously working.\\nThe problem is present for all people enabled for the VPN S2S with the customer and is blocking some project activities."
#     },
#     {
#         "title": "Request for network access to S3NET http://192.168.1.1:8080/s3netcm/S3netcmweb.htm",
#         "description": "Hi,\\n\\nI would like to request network access to this url http://192.168.1.2:8080/s3netcm/S3netcmweb.htm as I need to access the team's tickets\\n\\nThank you"
#     },
#     {
#         "title": "New channel creation",
#         "description": "Hi all,\\n\\nI need a new channel created for consultants where to share info and documents."
#     },
#     {
#         "title": "GFT ITALIA_mass upload request",
#         "description": "Hi,\\n\\nI need to do a lot of changes in Job Relationship related to the field DM.\\n\\nIn fact we decided to align the LM to DM for all our employees.\\n\\nIs it possible to do a mass upload?\\n\\nDoing everything by hand would take a long time.\\n\\nLet me know\\n\\nMany thanks\\n\\nRachel"
#     },
#     {
#         "title": "We need a new PROD virtual machine in Azure",
#         "description": "Because: https://short.link/path\\n\\nWe need a new Ubuntu LTS machines in Azure where migrate the SFTP services from TAMBOSFTP machines still located in STG,\\nThe original machine is exposed, no idea if there is a VPN tunnel, white list IP address or whatever.\\nThis machine is PROD environments, not DEV / BUSINESS, this is SAP 4 related machine\\n\\nStandars B2ms Azure machines size will be perfect."
#     },
#     {
#         "title": "I need a Jira license to access project ZXCVBN",
#         "description": "I can't see and edit ticket of ZXCVBN project. Everytime I click on a link (ex. [https://example.com](https://short.link)), the browser opens the Example Support Portal page (https://example.com/portals)"
#     },
#     {
#         "title": "Need administrative rights",
#         "description": "Need administrative rights to install software, setup and run the projects"
#     },
#     {
#         "title": "Need password reset for account and PC",
#         "description": "Hi\\nI've forgot to change my password on Friday, an today I've surpassed the limit time to change it\\n\\nTring to change and opening Citrix, the application tell me to change the password obligatory from app\\nAfter change it, i do the normal procedure to change, but not work\\nEvery time tell me i don't match the requirement password or history\\n\\nThe name of laptop is GFT-XE3Tumk5xol"
#     },
#     {
#         "title": "Install F5 VPN Client for BPER",
#         "description": "Hi,\\n\\nI need to install the F5 VPN Client to work on BPER customer's projects.\\n\\nThank you\\n\\nSalvatore"
#     },
#     {
#         "title": "Offboard employee Liam Castillo on 2024-07-16 [Ticket to manage virtual devices retirement]",
#         "description": "Initials: LICA\\nFirst Name: Liam\\nLast Name: Castillo\\nSF employee ID: 927361\\nLeaving date: 2024-07-16\\nCountry: ITA\\nLegal Entity: IT02\\nLocation: Padova\\nEmployee Level: L1 (Entry)\\nLine manager: Marco Esposito (Milano)"
#     },
#     {
#         "title": "I need Jira license",
#         "description": "I need Jira license to access on RXIURKPM"
#     },
#     {
#         "title": "I need a headphones",
#         "description": "Hi all,\\n\\nI have a problem with the headphones, they dont' work.\\n\\nCould you please change them?\\n\\nThanks in advance for your help,\\n\\nBest regards\\n\\nLeonardo"
#     },
#     {
#         "title": "Change permission for user",
#         "description": "I need to change permission on kimble for user jdoe\\n\\nPlease set the same permission of my user b123\\n\\nRegards,"
#     },
#     {
#         "title": "We need Jira license for Babylon Mutual project",
#         "description": "Hi,\\n\\nIn order to handle the project issues correctly, I kindly ask you to reassign Jira licences to the following people\\n\\nFull nameEmail\\nHugo Vasquezhugo.vasquez@randomcompany.com\\nLuca Ferreroluca.ferrero@randomcompany.com\\nMarco Leoramarco.leora@randomcompany.com\\nDiego Rovellidiego.rovelli@randomcompany.com\\nLorenzo Piagaloronzo.piaga@randomcompany.com\\n\\nRegards,\\n\\nOscar"
#     },
#     {
#         "title": "Need to install softwares in the descriptions for Academy Java",
#         "description": "Need administrator rights for 2 weeks"
#     },
#     {
#         "title": "Need support to reject an expense claim",
#         "description": "Hi All,\\n\\nwe want to to point out a problem with Alyssa Marlow's expense claim \\""EXC44183"" related to the engagement ""ABC Corp - General Administration 2024 (KE043899-001)"".\\n\\nThe expense is in ""pending Approval"" status, but the Approver, John Rossi, has left ABC Corp and we are unable to approve or reject it.\\n\\nWe have already fixed the element's approval rules by setting the new approver. Could you please reject that expense claim? Then Alyssa Marlow will submit the expense again.\\n\\nIf you need any further information, please just let me know.\\n\\nThank you.\\n\\nBest regards,\\n\\nChristina"
#     },
#     {
#         "title": "Please add colleague to Humaine Subscription",
#         "description": "please add Luigi Bianchi"
#     },
#     {
#         "title": "Need to install Ivanti 22.7r1.0-b28369-64bit",
#         "description": "I need to install the new version of Ivanti VPN otherwise I can't use the VPN for work without using Remote Desktop\\n\\nI need windows administrator permission to install\\n\\nThanks"
#     },
#     {
#         "title": "Request for Administrative Permissions for API Governance Tools",
#         "description": "Dear IT Team,\\n\\nAs part of our ongoing project, I require administrative permissions to install and configure specific API governance tools. These tools are critical for maintaining the efficiency, security, and reliability of our APIs (more details in justification).\\n\\nI kindly request your support in granting the necessary administrative permissions for these tools.\\n\\nThank you for your assistance."
#     },
#     {
#         "title": "Software installation for Java Academy",
#         "description": "Need administrator rights for java academy - 2 weeks"
#     },
#     {
#         "title": "Upgrade Ivanti Intesa VPN client",
#         "description": "Hi,\\n\\nI need to install a new version in order to upgrade the version of IVANTI VPN Intesa client\\n\\nI have new version on my laptop.\\n\\nthansk in advance\\n\\nPaolo"
#     },
#     {
#         "title": "gft_italia_new_dismiss_20240517_E-XJLG_573482",
#         "description": "Request forOFF - Boarding\\n\\nemployee datainternal / externalConsultant\\ndate of exit17 May 2024\\nSurnameROMANO\\nNameGIUSEPPE\\nprivate email address\\nadditional info\\nAccount information & permissionGFT codeE-XJLG\\n\\nVPN ACCESS (requires policy signature)YES\\nadditional permission\\nCompany informationCompanyGFT ITALIA SRL\\nRole\\nSAP code573482\\nLevel\\nClient unitISP\\nManagerROSSI\\nadditional info\\nGFT officeLocationTurin\\nworkplace required?\\n1°ST Day Location\\nRoom/Workplace\\nadditional info"
#     },
#     {
#         "title": "Substitution OLD Mobile Phone",
#         "description": "Ciao,\\n\\nQuesto ticket è stato creato per gestire la sostituzione del tuo smartphone che risulta obsoleto.\\n\\nSpediremo alla tua filiale di riferimento il nuovo telefono, sarai avvisato via mail. Contatta i servizi generali per organizzare il ritiro.\\n\\nRiceverai due notifiche automatiche dal sistema iTop :\\n\\n- relativo all' assegnazione del nuovo dispositivo (conferma la ricezione seguendo le istruzioni della mail)\\n- relativo alla riconsegna del vecchio, ricordati di restituire insieme tutti gli accessori\\n\\nPer la configurazione del nuovo dispositivo trovi tutte le informazioni necessarie a questi link:\\n\\nMFA: https://short.link1 e Teams/Outlook: https://short.link2\\n\\nCon l'occasione ti ricordiamo che hai la possibilità di riscattare il tuo vecchio smartphone. Per esercitare la prelazione è sufficiente confermarlo nel ticket.\\n\\nIl prezzo è di 130€ IVA compresa.\\n\\nNOTE: Ricorda che la prelazione è valida se:\\n\\n- Entro 7 giorni dalla data di creazione del presente ticket eserciti il tuo diritto di prelazione.\\n- Il vecchio smartphone viene riconsegnato all'ufficio IT entro e non oltre 30 giorni dalla data di creazione del presente ticket."
#     },
#     {
#         "title": "Need administrator rights",
#         "description": "Good Morning,\\n\\ncan i have administrator rights to install development software and change environments variables?\\n\\nThanks,\\n\\nAngela"
#     },
#     {
#         "title": "I need Azure DevOps extension free",
#         "description": "Good morning,\\n\\nWe are working on managing our projects on Azure DevOps and I would like to request the installation of this free extension available from the Azure MarketPlace:\\n\\nhttps://short.link/ABC123\\n\\nThe extension is called: Wiql Editor and should allow us to develop professional queries.\\n\\nI would like to activate it on the following project:\\n\\nhttps://short.link/XYZ789\\n\\nRegards,\\n\\nGiuseppe"
#     }
# ]

# Find these tickets in the original dataset and get their expected first replies

test_tickets = []
test_indices = []
used_indices = set()  # Track already used indices to prevent duplicates

print("🔍 Finding hardcoded test tickets in original dataset...")
for i, test_ticket in enumerate(hardcoded_test_tickets, 1):
    matched_row = None
    
    # Strategy 1: Try exact title match first
    exact_title_matches = df_original[df_original['Title_anon'].str.lower() == test_ticket['title'].lower()]
    if len(exact_title_matches) > 0:
        for _, row in exact_title_matches.iterrows():
            if row.name not in used_indices:
                matched_row = row
                break
    
    # Strategy 2: Try partial title match (more specific than before)
    if matched_row is None:
        # Use more specific matching - require at least 50% of title to match
        min_match_length = max(len(test_ticket['title']) // 2, 10)
        title_pattern = test_ticket['title'][:min_match_length]
        
        title_matches = df_original[df_original['Title_anon'].str.contains(
            re.escape(title_pattern), case=False, na=False)]
        
        if len(title_matches) > 0:
            for _, row in title_matches.iterrows():
                if row.name not in used_indices:
                    matched_row = row
                    break
    
    # Strategy 3: Try description-based matching with higher threshold
    if matched_row is None:
        # Use longer description pattern and check for uniqueness
        min_desc_length = max(len(test_ticket['description']) // 3, 50)
        desc_pattern = test_ticket['description'][:min_desc_length]
        
        desc_matches = df_original[df_original['Description_anon'].str.contains(
            re.escape(desc_pattern), case=False, na=False)]
        
        if len(desc_matches) > 0:
            for _, row in desc_matches.iterrows():
                if row.name not in used_indices:
                    matched_row = row
                    break
    
    # Strategy 4: For onboarding tickets, use specific patterns
    if matched_row is None and 'gft italia_new_entry' in test_ticket['title'].lower():
        onboarding_pattern = r'gft.*italia.*new.*entry.*\d{8}'
        onboarding_matches = df_original[df_original['Title_anon'].str.contains(
            onboarding_pattern, case=False, na=False, regex=True)]
        
        if len(onboarding_matches) > 0:
            for _, row in onboarding_matches.iterrows():
                if row.name not in used_indices:
                    matched_row = row
                    break
    
    # Add matched ticket if found
    if matched_row is not None:
        test_tickets.append({
            "title": matched_row["Title_anon"],
            "description": matched_row["Description_anon"], 
            "expected_first_reply": matched_row["first_reply"],
            "original_index": matched_row.name
        })
        test_indices.append(matched_row.name)
        used_indices.add(matched_row.name)  # Mark as used
        print(f"✅ {i:2d}. Found: {matched_row['Title_anon'][:50]}... (Index: {matched_row.name})")
    else:
        print(f"❌ {i:2d}. Not found: {test_ticket['title'][:50]}...")

print(f"\n📋 Successfully matched {len(test_tickets)} unique test tickets")
print(f"📋 Unique indices used: {len(set(test_indices))}")

# Verify no duplicates
if len(test_indices) != len(set(test_indices)):
    print("⚠️ WARNING: Duplicate indices detected!")
    duplicates = [idx for idx in test_indices if test_indices.count(idx) > 1]
    print(f"Duplicate indices: {set(duplicates)}")


# Create a copy of the dataset WITHOUT the test tickets
print("💾 Creating knowledge base copy without test tickets...")
df_copy = df_original.drop(test_indices).reset_index(drop=True)
knowledge_base_path = "tickets_large_first_reply_label_copy.csv"
df_copy.to_csv(knowledge_base_path, index=False)

print(f"✅ Knowledge base copy created:")
print(f"   Original dataset: {len(df_original)} tickets")
print(f"   Copy dataset: {len(df_copy)} tickets")
print(f"   Removed for testing: {len(test_indices)} tickets")

# Pre-build the RAG system ONCE (this will use cache on subsequent runs)
print("🚀 Pre-building RAG system (will use cache if available)...")
rag_system = get_or_create_rag_system(knowledge_base_path)

# Initialize Sentence-BERT models for evaluation
print("🔄 Loading evaluation models...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # Original model
sbert_model = SentenceTransformer("all-mpnet-base-v2")    # Sentence-BERT specific model

def process_ticket_with_cached_rag(ticket_title, ticket_description, rag_system):
    """Process ticket using pre-built RAG system"""
    from resolution_task import generate_response
    
    result = generate_response(ticket_title, ticket_description, rag_system)
    return result

# Add this function after the existing imports and before the test_rag_system function

# Add this function after the existing imports and before the test_rag_system function

def get_expected_team_from_original_dataset(original_index, df_original):
    """
    Get the expected team for a ticket based on its actual CSV assignment.
    NO MAPPING - return the actual CSV team since model was trained on CSV data.
    """
    try:
        row = df_original.loc[original_index]
        
        # Get the actual team from the CSV data - prefer 'Last team ID->Name' as it's the final assignment
        csv_team = str(row.get('Last team ID->Name', '')).strip()
        
        if not csv_team or csv_team == 'nan' or csv_team == '' or csv_team == 'None':
            # If no team in Last team ID->Name, try 'Team->Name' 
            csv_team = str(row.get('Team->Name', '')).strip()
        
        if not csv_team or csv_team == 'nan' or csv_team == '' or csv_team == 'None':
            print(f"⚠️ No team found for ticket index {original_index}, using Service Desk")
            return "(GI-SM) Service Desk"  # Default fallback
        
        # Return the actual CSV team - no mapping needed since model was trained on CSV data
        return csv_team
            
    except Exception as e:
        print(f"⚠️ Error getting expected team for index {original_index}: {e}")
        return "(GI-SM) Service Desk"  # Default fallback


def calculate_team_accuracy(results):
    """Calculate team prediction accuracy"""
    correct_predictions = 0
    total_predictions = 0
    team_confusion_matrix = {}
    
    for result in results:
        if 'error' not in result and 'predicted_team' in result:
            expected_team = result.get('expected_team', 'Unknown')
            predicted_team = result.get('predicted_team', 'Unknown')
            
            total_predictions += 1
            if expected_team == predicted_team:
                correct_predictions += 1
            
            # Build confusion matrix
            if expected_team not in team_confusion_matrix:
                team_confusion_matrix[expected_team] = {}
            if predicted_team not in team_confusion_matrix[expected_team]:
                team_confusion_matrix[expected_team][predicted_team] = 0
            team_confusion_matrix[expected_team][predicted_team] += 1
    
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    return accuracy, team_confusion_matrix

def print_team_analysis(results):
    """Print detailed team prediction analysis"""
    print(f"\n👥 TEAM PREDICTION ANALYSIS:")
    print("="*80)
    
    accuracy, confusion_matrix = calculate_team_accuracy(results)
    print(f"Overall Team Prediction Accuracy: {accuracy:.1f}%")
    
    print(f"\n📊 Team Prediction Details:")
    print("-"*80)
    for result in results:
        if 'error' not in result:
            expected = result.get('expected_team', 'Unknown')
            predicted = result.get('predicted_team', 'Unknown')
            match_status = "✅" if expected == predicted else "❌"
            
            print(f"{match_status} Ticket: {result['title'][:40]}...")
            print(f"    Expected Team: {expected}")
            print(f"    Predicted Team: {predicted}")
            print(f"    Similarity Score: {result.get('cosine_similarity', 0):.4f}")
            print()
    
    # Print confusion matrix
    if confusion_matrix:
        print(f"\n📋 Team Prediction Confusion Matrix:")
        print("-"*80)
        all_teams = set()
        for expected, predictions in confusion_matrix.items():
            all_teams.add(expected)
            all_teams.update(predictions.keys())
        
        for expected_team in sorted(all_teams):
            if expected_team in confusion_matrix:
                print(f"\nExpected: {expected_team}")
                for predicted_team in sorted(all_teams):
                    count = confusion_matrix[expected_team].get(predicted_team, 0)
                    if count > 0:
                        print(f"  → Predicted as {predicted_team}: {count}")
    
    return accuracy, confusion_matrix



# Update the test_rag_system function to include team accuracy evaluation

def test_rag_system():
    """Test the RAG system with hardcoded tickets using cached embeddings"""
    results = []
    cosine_similarities = []
    sbert_cosine_similarities = []

    print(f"\n🧪 Testing RAG system with {len(test_tickets)} hardcoded test tickets")
    print("="*80)

    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n{'='*70}")
        print(f"🎫 Processing Ticket {i}/{len(test_tickets)}")
        print(f"Title: {ticket['title']}")
        print(f"{'='*70}")
        print(f"📝 Description: {ticket['description'][:100]}...")
        
        try:
            # Get expected team based on heuristics
            expected_team = get_expected_team_from_original_dataset(ticket["original_index"], df_original)
            
            # Process the ticket using the pre-built RAG system (NO REBUILDING!)
            result = process_ticket_with_cached_rag(ticket["title"], ticket["description"], rag_system)
            
            print(f"📋 Predicted Class: {result['classification']}")
            print(f"👥 Expected Team: {expected_team}")
            print(f"👥 Predicted Team: {result['predicted_team']}")
            
            # Check team prediction accuracy
            team_match = "✅" if expected_team == result['predicted_team'] else "❌"
            print(f"🎯 Team Prediction: {team_match}")
            
            print(f"🤖 Generated Response: {result['response'][:200]}...")
            print(f"✅ Expected (Original): {ticket['expected_first_reply'][:200]}...")

            # Compute cosine similarity using all-MiniLM-L6-v2 (original)
            expected_embedding = sentence_model.encode(ticket["expected_first_reply"], convert_to_tensor=True)
            generated_embedding = sentence_model.encode(result["response"], convert_to_tensor=True)
            cosine_similarity = util.cos_sim(expected_embedding, generated_embedding).item()
            cosine_similarities.append(cosine_similarity)
            
            # Compute sentence-BERT cosine similarity using all-mpnet-base-v2
            expected_sbert_embedding = sbert_model.encode(ticket["expected_first_reply"], convert_to_tensor=True)
            generated_sbert_embedding = sbert_model.encode(result["response"], convert_to_tensor=True)
            sbert_cosine_similarity = util.cos_sim(expected_sbert_embedding, generated_sbert_embedding).item()
            sbert_cosine_similarities.append(sbert_cosine_similarity)
            
            print(f"📊 Cosine Similarity (MiniLM): {cosine_similarity:.4f}")
            print(f"📊 Sentence-BERT Similarity (MPNet): {sbert_cosine_similarity:.4f}")

            results.append({
                "ticket_number": i,
                "title": ticket["title"],
                "description": ticket["description"],
                "expected_first_reply": ticket["expected_first_reply"],
                "generated_response": result["response"],
                "predicted_class": result["classification"],
                "expected_team": expected_team,  # NEW: Expected team
                "predicted_team": result["predicted_team"],
                "team_prediction_correct": expected_team == result["predicted_team"],  # NEW: Team accuracy
                "cosine_similarity": cosine_similarity,
                "sbert_cosine_similarity": sbert_cosine_similarity,  # NEW: Sentence-BERT similarity
                "original_index": ticket["original_index"]
            })
            
        except Exception as e:
            print(f"❌ Error processing ticket {i}: {e}")
            results.append({
                "ticket_number": i,
                "title": ticket["title"],
                "description": ticket["description"],
                "error": str(e),
                "cosine_similarity": 0.0,
                "sbert_cosine_similarity": 0.0,  # NEW: Include in error case
                "team_prediction_correct": False,
                "original_index": ticket["original_index"]
            })

    # Calculate statistics including team accuracy
    if cosine_similarities:
        avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
        max_similarity = max(cosine_similarities)
        min_similarity = min(cosine_similarities)
        
        # Calculate sentence-BERT statistics
        avg_sbert_similarity = sum(sbert_cosine_similarities) / len(sbert_cosine_similarities)
        max_sbert_similarity = max(sbert_cosine_similarities)
        min_sbert_similarity = min(sbert_cosine_similarities)
        
        print(f"\n📊 FINAL EVALUATION RESULTS:")
        print(f"="*50)
        print(f"Total tickets processed: {len(test_tickets)}")
        print(f"Successful evaluations: {len(cosine_similarities)}")
        print(f"\n🔍 Cosine Similarity (MiniLM-L6-v2):")
        print(f"Average: {avg_cosine_similarity:.4f}")
        print(f"Best Match: {max_similarity:.4f}")
        print(f"Worst Match: {min_similarity:.4f}")
        print(f"\n🎯 Sentence-BERT Similarity (MPNet-base-v2):")
        print(f"Average: {avg_sbert_similarity:.4f}")
        print(f"Best Match: {max_sbert_similarity:.4f}")
        print(f"Worst Match: {min_sbert_similarity:.4f}")
        
        # Show distribution for both similarity measures
        high_quality = sum(1 for s in cosine_similarities if s > 0.7)
        medium_quality = sum(1 for s in cosine_similarities if 0.4 <= s <= 0.7)
        low_quality = sum(1 for s in cosine_similarities if s < 0.4)
        
        sbert_high_quality = sum(1 for s in sbert_cosine_similarities if s > 0.7)
        sbert_medium_quality = sum(1 for s in sbert_cosine_similarities if 0.4 <= s <= 0.7)
        sbert_low_quality = sum(1 for s in sbert_cosine_similarities if s < 0.4)
        
        print(f"\n📈 Quality Distribution (MiniLM-L6-v2):")
        print(f"High similarity (>0.7): {high_quality} tickets ({high_quality/len(cosine_similarities)*100:.1f}%)")
        print(f"Medium similarity (0.4-0.7): {medium_quality} tickets ({medium_quality/len(cosine_similarities)*100:.1f}%)")
        print(f"Low similarity (<0.4): {low_quality} tickets ({low_quality/len(cosine_similarities)*100:.1f}%)")
        
        print(f"\n📈 Quality Distribution (MPNet-base-v2):")
        print(f"High similarity (>0.7): {sbert_high_quality} tickets ({sbert_high_quality/len(sbert_cosine_similarities)*100:.1f}%)")
        print(f"Medium similarity (0.4-0.7): {sbert_medium_quality} tickets ({sbert_medium_quality/len(sbert_cosine_similarities)*100:.1f}%)")
        print(f"Low similarity (<0.4): {sbert_low_quality} tickets ({sbert_low_quality/len(sbert_cosine_similarities)*100:.1f}%)")
        
        # NEW: Team accuracy analysis
        team_accuracy, confusion_matrix = print_team_analysis(results)

    return results


# Add this function after the existing imports and before the test_rag_system function

def evaluate_ticket_classifier_accuracy(test_tickets, sample_size=None):
    """Simple accuracy evaluation of the ticket classifier model"""
    
    print(f"\n🧪 TICKET CLASSIFIER ACCURACY EVALUATION")
    print("="*60)
    
    from resolution_task import classify_ticket
    
    # Use all test tickets or sample
    tickets_to_test = test_tickets[:sample_size] if sample_size else test_tickets
    
    print(f"📊 Testing classifier on {len(tickets_to_test)} tickets")
    
    # Define expected classifications based on ticket content
    def get_expected_classification(title, description):
        """Get expected classification based on ticket content"""
        combined_text = f"{title} {description}".lower()
        
        if any(word in combined_text for word in ['admin', 'administrator', 'rights', 'privileges']):
            return "admin_rights"
        elif any(word in combined_text for word in ['vpn', 'tunnel', 'remote access']):
            return "vpn_request"
        elif any(word in combined_text for word in ['onboard', 'new employee', 'employee setup', 'gft italia_new_entry']):
            return "onboarding"
        elif any(word in combined_text for word in ['badge', 'access card', 'building access']):
            return "badge_access"
        elif any(word in combined_text for word in ['mailbox', 'email', 'outlook']):
            return "email_support"
        elif any(word in combined_text for word in ['software', 'install', 'installation']):
            return "software_request"
        elif any(word in combined_text for word in ['network', 'switch', 'port']):
            return "network_support"
        elif any(word in combined_text for word in ['invoice', 'sap', 'erp']):
            return "other"  # Financial system issues often classified as "other"
        else:
            return "other"
    
    # Evaluate each ticket
    correct_predictions = 0
    total_predictions = 0
    classification_results = []
    
    print("🔄 Making predictions...")
    
    for i, ticket in enumerate(tickets_to_test, 1):
        try:
            # Get expected classification
            expected_class = get_expected_classification(ticket["title"], ticket["description"])
            
            # Get actual prediction
            ticket_text = f"{ticket['title']} {ticket['description']}"
            predicted_class, template = classify_ticket(ticket_text)
            
            # Check accuracy
            is_correct = predicted_class == expected_class
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            classification_results.append({
                "ticket_number": i,
                "title": ticket["title"][:40] + "...",
                "expected": expected_class,
                "predicted": predicted_class,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"❌ Error classifying ticket {i}: {e}")
            total_predictions += 1
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Print results
    print(f"\n📈 CLASSIFIER ACCURACY RESULTS:")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    # Show per-class breakdown
    class_stats = {}
    for result in classification_results:
        expected = result["expected"]
        predicted = result["predicted"]
        
        if expected not in class_stats:
            class_stats[expected] = {"total": 0, "correct": 0}
        
        class_stats[expected]["total"] += 1
        if result["correct"]:
            class_stats[expected]["correct"] += 1
    
    print(f"\n📊 Per-Class Accuracy:")
    print("-"*50)
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        class_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"{class_name:20} {class_acc:6.1f}% ({stats['correct']:2d}/{stats['total']:2d})")
    
    # Show some examples
    print(f"\n🔍 SAMPLE CLASSIFICATIONS:")
    print("-"*50)
    for result in classification_results[:5]:  # Show first 5
        status = "✅" if result["correct"] else "❌"
        print(f"{status} {result['title']}")
        print(f"    Expected: {result['expected']} | Predicted: {result['predicted']}")
        print()
    
    return accuracy, classification_results


def evaluate_retrieval_component(test_tickets, knowledge_base_path):
    """
    Evaluate just the retrieval component of the RAG system
    
    Args:
        test_tickets: List of test tickets
        knowledge_base_path: Path to knowledge base
    
    Returns:
        Dict with retrieval evaluation metrics
    """
    print("🔍 Evaluating RAG Retrieval Component...")
    
    # Load RAG system
    from resolution_task import load_knowledge_base, RAGSystem
    
    knowledge_base = load_knowledge_base(knowledge_base_path)
    rag_system = RAGSystem(knowledge_base)
    rag_system.build_index()
    
    # Initialize evaluation model
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    retrieval_results = []
    
    for i, ticket in enumerate(test_tickets):
        print(f"  Evaluating retrieval {i+1}/{len(test_tickets)}: {ticket['title'][:50]}...")
        
        query_text = f"{ticket['title']} {ticket['description']}"
        
        # Retrieve similar documents with different top_k values
        for top_k in [3, 5, 10]:
            start_time = time.time()
            
            retrieved_docs = rag_system.retrieve_similar_replies(
                query_text, top_k=top_k
            )
            
            retrieval_time = time.time() - start_time
            
            # Evaluate retrieval quality
            retrieval_quality = evaluate_retrieval_relevance(
                ticket, retrieved_docs, eval_model
            )
            
            # Calculate diversity
            diversity = calculate_retrieval_diversity_simple(retrieved_docs)
            
            # Calculate semantic similarity with expected reply
            semantic_sim = calculate_max_semantic_similarity(
                ticket['expected_first_reply'], retrieved_docs, eval_model
            )
            
            retrieval_results.append({
                'ticket_title': ticket['title'],
                'top_k': top_k,
                'retrieval_quality': retrieval_quality,
                'diversity': diversity,
                'semantic_similarity': semantic_sim,
                'retrieval_time': retrieval_time,
                'num_retrieved': len(retrieved_docs)
            })
    
    # Calculate summary statistics
    summary_by_k = {}
    for k in [3, 5, 10]:
        k_results = [r for r in retrieval_results if r['top_k'] == k]
        summary_by_k[k] = {
            'avg_retrieval_quality': np.mean([r['retrieval_quality'] for r in k_results]),
            'avg_diversity': np.mean([r['diversity'] for r in k_results]),
            'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in k_results]),
            'avg_retrieval_time': np.mean([r['retrieval_time'] for r in k_results])
        }
    
    print("\n" + "="*50)
    print("RETRIEVAL COMPONENT EVALUATION")
    print("="*50)
    
    for k in [3, 5, 10]:
        stats = summary_by_k[k]
        print(f"\nTop-{k} Results:")
        print(f"  Avg Retrieval Quality: {stats['avg_retrieval_quality']:.3f}")
        print(f"  Avg Diversity: {stats['avg_diversity']:.3f}")
        print(f"  Avg Semantic Similarity: {stats['avg_semantic_similarity']:.3f}")
        print(f"  Avg Retrieval Time: {stats['avg_retrieval_time']:.4f}s")
    
    return {
        'summary_by_k': summary_by_k,
        'detailed_results': retrieval_results
    }

def evaluate_retrieval_relevance(ticket, retrieved_docs, eval_model):
    """
    Evaluate relevance of retrieved documents to the query
    
    Args:
        ticket: Test ticket dict
        retrieved_docs: Retrieved documents DataFrame
        eval_model: SentenceTransformer model
    
    Returns:
        Float: Relevance score (0-1)
    """
    if retrieved_docs.empty:
        return 0.0
    
    query_text = f"{ticket['title']} {ticket['description']}"
    query_emb = eval_model.encode([query_text])
    
    relevance_scores = []
    
    for _, doc in retrieved_docs.iterrows():
        doc_text = f"{doc.get('Title_anon', '')} {doc.get('Description_anon', '')}"
        doc_emb = eval_model.encode([doc_text])
        
        # Calculate semantic similarity
        similarity = cosine_similarity(query_emb, doc_emb)[0][0]
        relevance_scores.append(max(0, similarity))
    
    # Use DCG-like scoring (higher weight for top results)
    weights = [1.0 / np.log2(i + 2) for i in range(len(relevance_scores))]
    weighted_relevance = np.average(relevance_scores, weights=weights)
    
    return weighted_relevance

def calculate_retrieval_diversity_simple(retrieved_docs):
    """
    Calculate diversity of retrieved documents (simplified version)
    
    Args:
        retrieved_docs: DataFrame with retrieved documents
    
    Returns:
        Float: Diversity score (0-1)
    """
    if len(retrieved_docs) < 2:
        return 1.0
    
    # Simple diversity based on unique titles
    unique_titles = retrieved_docs['Title_anon'].nunique()
    total_docs = len(retrieved_docs)
    
    title_diversity = unique_titles / total_docs
    
    # Add category diversity if available
    category_diversity = 1.0
    if 'label_auto' in retrieved_docs.columns:
        unique_categories = retrieved_docs['label_auto'].nunique()
        category_diversity = unique_categories / len(retrieved_docs)
    
    return (title_diversity + category_diversity) / 2

def calculate_max_semantic_similarity(expected_reply, retrieved_docs, eval_model):
    """
    Calculate maximum semantic similarity between expected reply and retrieved replies
    
    Args:
        expected_reply: Expected first reply text
        retrieved_docs: DataFrame with retrieved documents
        eval_model: SentenceTransformer model
    
    Returns:
        Float: Maximum similarity score
    """
    if pd.isna(expected_reply) or retrieved_docs.empty:
        return 0.0
    
    expected_emb = eval_model.encode([expected_reply])
    max_similarity = 0.0
    
    for _, doc in retrieved_docs.iterrows():
        first_reply = doc.get('first_reply', '')
        if pd.notna(first_reply) and str(first_reply).strip():
            reply_emb = eval_model.encode([str(first_reply)])
            similarity = cosine_similarity(expected_emb, reply_emb)[0][0]
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity


# Update the if __name__ == "__main__": section at the bottom:

if __name__ == "__main__":
    # Run the RAG system evaluation
    print("🚀 Starting RAG System Evaluation...")
    test_results = test_rag_system()
    
    # Save the results to a file for further analysis
    output_file = "test_results_cached.json"
    results_df = pd.DataFrame(test_results)
    results_df.to_json(output_file, orient="records", indent=4)
    print(f"\n💾 Detailed results saved to {output_file}")
    
    # Also save as CSV for easier analysis
    csv_file = "test_results_cached.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"💾 Results also saved as CSV: {csv_file}")
    
    # NEW: Run ticket classifier accuracy evaluation
    print(f"\n" + "="*80)
    print("TICKET CLASSIFIER EVALUATION")
    print("="*80)
    
    classifier_accuracy, classification_results = evaluate_ticket_classifier_accuracy(
        test_tickets, 
        sample_size=len(test_tickets)  # Test on all tickets
    )
    
    # NEW: Run retrieval-only evaluation
    # print(f"\n" + "="*80)
    # print("RETRIEVAL COMPONENT EVALUATION")
    # print("="*80)
    
    # retrieval_results = evaluate_retrieval_component(test_tickets, knowledge_base_path)
    
    # # Save retrieval results
    # retrieval_output_file = "retrieval_evaluation_results.json"
    # with open(retrieval_output_file, "w") as f:
    #     json.dump(retrieval_results, f, indent=4, default=str)
    # print(f"💾 Retrieval evaluation saved to {retrieval_output_file}")
    
    # Show detailed comparison for best and worst results
    successful_results = [r for r in test_results if 'error' not in r]
    if successful_results:
        print(f"\n📋 DETAILED COMPARISON EXAMPLES:")
        print("="*80)
        
        # Show best performing ticket
        best_result = max(successful_results, key=lambda x: x['cosine_similarity'])
        print(f"\n🏆 BEST PERFORMING TICKET (Similarity: {best_result['cosine_similarity']:.4f}):")
        print(f"Title: {best_result['title']}")
        print(f"Expected Team: {best_result.get('expected_team', 'N/A')}")
        print(f"Predicted Team: {best_result.get('predicted_team', 'N/A')}")
        print(f"Team Prediction: {'✅ Correct' if best_result.get('team_prediction_correct', False) else '❌ Incorrect'}")
        print(f"Expected: {best_result['expected_first_reply'][:300]}...")
        print(f"Generated: {best_result['generated_response'][:300]}...")
        
        # Show worst performing ticket
        worst_result = min(successful_results, key=lambda x: x['cosine_similarity'])
        print(f"\n⚠️ WORST PERFORMING TICKET (Similarity: {worst_result['cosine_similarity']:.4f}):")
        print(f"Title: {worst_result['title']}")
        print(f"Expected Team: {worst_result.get('expected_team', 'N/A')}")
        print(f"Predicted Team: {worst_result.get('predicted_team', 'N/A')}")
        print(f"Team Prediction: {'✅ Correct' if worst_result.get('team_prediction_correct', False) else '❌ Incorrect'}")
        print(f"Expected: {worst_result['expected_first_reply'][:300]}...")
        print(f"Generated: {worst_result['generated_response'][:300]}...")
        
        # Overall performance summary including classifier accuracy
        team_correct = sum(1 for r in successful_results if r.get('team_prediction_correct', False))
        team_accuracy = (team_correct / len(successful_results)) * 100 if successful_results else 0
        response_quality = sum(r['cosine_similarity'] for r in successful_results) / len(successful_results)
        
        print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
        print("="*60)
        print(f"Ticket Classification Accuracy: {classifier_accuracy:.1f}%")
        print(f"Team Prediction Accuracy: {team_accuracy:.1f}% ({team_correct}/{len(successful_results)})")
        print(f"Response Quality (Cosine Similarity): {response_quality:.4f}")
        
        # Add retrieval performance summary
        # print(f"\n🔍 RETRIEVAL PERFORMANCE SUMMARY:")
        # print("-"*60)
        # for k in [3, 5, 10]:
        #     stats = retrieval_results['summary_by_k'][k]
        #     print(f"Top-{k} Retrieval Quality: {stats['avg_retrieval_quality']:.3f}")
        #     print(f"Top-{k} Diversity: {stats['avg_diversity']:.3f}")
        #     print(f"Top-{k} Semantic Similarity: {stats['avg_semantic_similarity']:.3f}")
        #     print(f"Top-{k} Avg Time: {stats['avg_retrieval_time']:.4f}s")
        #     print()
        
        print(f"\n🎯 SYSTEM PERFORMANCE BREAKDOWN:")
        print("-"*60)
        print(f"1. Ticket Classification: {classifier_accuracy:.1f}% - How well the system identifies request types")
        print(f"2. Team Assignment: {team_accuracy:.1f}% - How well the system assigns to correct teams")
        print(f"3. Response Generation: {response_quality:.4f} - How similar generated responses are to expected ones")
        # print(f"4. Retrieval Quality: {retrieval_results['summary_by_k'][5]['avg_retrieval_quality']:.3f} - How well the system retrieves relevant documents")
        # print(f"5. Retrieval Diversity: {retrieval_results['summary_by_k'][5]['avg_diversity']:.3f} - How diverse the retrieved results are")
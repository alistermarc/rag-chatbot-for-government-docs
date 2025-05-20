import os
import time
import boto3
import pandas as pd
from collections import defaultdict
from openai import OpenAI
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from django.conf import settings
import weaviate
import re

from .models import ModelManager

# Initialize models 
base_dir = os.path.dirname(os.path.abspath(__file__))
embedding_path = os.path.join(base_dir, "..", "embeddings")
# For embedding model
model_manager = ModelManager(embedding_path)  
# For classification, query transformation, generation, and validation
client = OpenAI()

class TextractProcessor:
    def __init__(self, base_dir, bucket_name=None):
        self.textract = boto3.client('textract')
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name or settings.AWS_BUCKET_NAME
        self.base_dir = base_dir 

    def process_pdf(self, file_path):
        file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        output_folder = os.path.join(self.base_dir, file_name_no_ext)
        os.makedirs(output_folder, exist_ok=True)

        # Upload to S3
        s3_key = f'{os.path.basename(self.base_dir)}/{os.path.basename(file_path)}'
        self.s3.upload_file(file_path, self.bucket_name, s3_key)
        print(f"Uploaded to S3: {s3_key}")

        # Start Textract job
        response = self.textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': self.bucket_name, 'Name': s3_key}},
            FeatureTypes=['LAYOUT', 'TABLES']
        )
        job_id = response['JobId']
        print(f"Textract Job Started: {job_id}")

        # Poll for completion
        while True:
            result = self.textract.get_document_analysis(JobId=job_id)
            status = result['JobStatus']
            print(f"Job status: {status}")
            if status in ['SUCCEEDED', 'FAILED']:
                if status == 'FAILED':
                    raise Exception("Textract job failed.")
                break
            time.sleep(5)

        # Retrieve all results
        results = []
        next_token = None
        while True:
            response = self.textract.get_document_analysis(JobId=job_id, NextToken=next_token) if next_token else self.textract.get_document_analysis(JobId=job_id)
            results.append(response)
            next_token = response.get('NextToken')
            if not next_token:
                break

        # Save results to output folder
        self.save_layout(results, output_folder)
        self.save_tables(results, output_folder)

    def save_layout(self, results, output_folder):
        rows = []
        layout_counters = defaultdict(int)
        reading_order = 0
        line_map = {}
        for page in results:
            line_map.update({b['Id']: b for b in page['Blocks'] if b['BlockType'] == 'LINE'})
            layout_blocks = [b for b in page['Blocks'] if b['BlockType'].startswith('LAYOUT')]

            for block in layout_blocks:
                layout_key = block['BlockType'].replace('LAYOUT_', '').capitalize()
                layout_counters[layout_key] += 1
                layout_label = f"{layout_key} {layout_counters[layout_key]}"

                line_text = ''
                for rel in block.get('Relationships', []):
                    if rel.get('Type') == 'CHILD':
                        line_text = ' '.join(line_map.get(i, {}).get('Text', '') for i in rel.get('Ids', []) if i in line_map)

                rows.append({
                    'Page number': block.get('Page', 1),
                    'Layout': layout_label,
                    'Text': line_text.strip(),
                    'Reading Order': reading_order,
                    'Confidence score % (Layout)': block['Confidence']
                })
                reading_order += 1

        layout_path = os.path.join(output_folder, 'layout.csv')
        pd.DataFrame(rows).to_csv(layout_path, index=False)
        print(f"Saved layout to {layout_path}")

    def get_text(self, cell, blocks_map):
        text = ''
        for rel in cell.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                for cid in rel['Ids']:
                    word = blocks_map.get(cid, {})
                    if word['BlockType'] == 'WORD':
                        t = word['Text']
                        text += f'"{t}" ' if "," in t and t.replace(",", "").isnumeric() else f"{t} "
                    elif word['BlockType'] == 'SELECTION_ELEMENT' and word.get('SelectionStatus') == 'SELECTED':
                        text += 'X '
        return text.strip()

    def get_table_rows(self, table, blocks_map):
        rows, scores = {}, []
        for rel in table.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                for cid in rel['Ids']:
                    cell = blocks_map.get(cid)
                    if cell and cell['BlockType'] == 'CELL':
                        row, col = cell['RowIndex'], cell['ColumnIndex']
                        rows.setdefault(row, {})[col] = self.get_text(cell, blocks_map)
                        scores.append(str(cell['Confidence']))
        return rows, scores

    def save_tables(self, results, output_folder):
        table_index = 1
        for page in results:
            blocks = page['Blocks']
            blocks_map = {b['Id']: b for b in blocks}
            table_blocks = [b for b in blocks if b['BlockType'] == 'TABLE']
            for table in table_blocks:
                rows, scores = self.get_table_rows(table, blocks_map)
                table_id = f"Table_{table_index}"
                table_csv = f"Table: {table_id}\n\n"

                col_count = 0
                for row in sorted(rows):
                    cols = rows[row]
                    col_count = len(cols)
                    table_csv += ','.join(cols[col] for col in sorted(cols)) + '\n'

                table_csv += '\n\n Confidence Scores % (Table Cell) \n'
                for i, score in enumerate(scores, 1):
                    table_csv += score + (',' if i % col_count else '\n')

                output_file = os.path.join(output_folder, f'table-{table_index}.csv')
                with open(output_file, 'w') as f:
                    f.write(table_csv)
                print(f"Saved: {output_file}")
                table_index += 1


class DocumentPreprocessor:
    def __init__(self, openai_model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = openai_model

    def summarize(self, output_folder):
        df = pd.read_csv(os.path.join(output_folder, "layout.csv"), encoding="utf-8")

        # Filter relevant rows
        filtered = df[(df['Layout'].str.contains('Section header|Section_header|Table')) |
                      ((df['Layout'].str.contains('Text|Footer')) & (df['Text'].str.len() > 5)) |
                      ((df['Layout'].str.contains('Header|Title')) &
                       (df['Text'].str.contains('AN ORDINANCE|WHEREAS|NOW, THEREFORE|ENACTED:', regex=True)))].copy()
        filtered = filtered.reset_index(drop=True)

        # Identify tables
        table_df = filtered[filtered['Layout'].str.startswith("Table")].copy()
        final_path = os.path.join(output_folder, "layout_final.csv")
        if table_df.empty:
            filtered.to_csv(final_path, index=False, encoding='utf-8')
            return final_path, filtered

        table_df['TableNumber'] = table_df['Layout'].str.extract(r'(\d+)').astype(int)
        table_groups = self._group_consecutive_tables(table_df)

        for group in table_groups:
            merged_text = self._merge_tables(output_folder, group)
            if not merged_text:
                continue

            kv_text = self._convert_to_key_value(merged_text)
            idx_main = filtered[filtered['Layout'] == f"Table {group[0]}"].index[0]
            filtered.loc[idx_main, 'Text'] = kv_text

            context = filtered.loc[idx_main - 1, 'Text'] if idx_main > 0 else ""
            summary = self._summarize_table(kv_text, context)

            summary_row = pd.DataFrame([{
                'Page number': filtered.loc[idx_main, 'Page number'],
                'Layout': f"Summary Table {group[0]}",
                'Text': summary,
                'Reading Order': filtered.loc[idx_main, 'Reading Order'],
                'Confidence score % (Layout)': "N/A"
            }])
            filtered = pd.concat([
                filtered.iloc[:idx_main + 1],
                summary_row,
                filtered.iloc[idx_main + 1:]
            ], ignore_index=True)

            for t in group[1:]:
                filtered = filtered[filtered['Layout'] != f"Table {t}"]

            with open(os.path.join(output_folder, f"merged_table_{group[0]}.csv"), "w", encoding="utf-8") as f:
                f.write(kv_text)

        # Save final output
        filtered.to_csv(final_path, index=False, encoding='utf-8')
        return final_path, filtered

    def _group_consecutive_tables(self, df):
        idxs = df.index.tolist()
        groups, group = [], [idxs[0]]
        for i in range(1, len(idxs)):
            if idxs[i] == idxs[i - 1] + 1:
                group.append(idxs[i])
            else:
                groups.append(group)
                group = [idxs[i]]
        groups.append(group)
        return [[df.loc[i, 'TableNumber'] for i in g] for g in groups]

    def _read_and_clean_table(self, folder, table_index):
        try:
            with open(os.path.join(folder, f'table-{table_index}.csv'), 'r', encoding='utf-8') as f:
                content = f.read()
                return "\n".join(content.split("Confidence Scores % (Table Cell)")[0].strip().split("\n")[:-1])
        except FileNotFoundError:
            return None

    def _merge_tables(self, folder, table_group):
        lines = []
        for idx in table_group:
            content = self._read_and_clean_table(folder, idx)
            if content:
                lines += [line.rstrip(',').replace(' ,', ',').replace(', ', ',') for line in content.split("\n") if line.strip()]
        return "\n".join(lines)

    def _convert_to_key_value(self, table_text):
        prompt = f"Convert the following CSV table into key-value pairs:\n\n{table_text}\n\nReturn answer only."
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Conversion failed: {e}"

    def _summarize_table(self, kv_text, context):
        prompt = f"Summarize the information in this table into one or two concise sentences using the provided context, without using a structured format.\n\nContext: {context}\n\nTable Data:\n{kv_text}\n\nReturn only the summary."
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summary failed: {e}"


class DocumentIndexer:
    def __init__(self, model_manager=model_manager):
        self.model_manager = model_manager
        self.client = weaviate.connect_to_local()

        self.section_categories = {
            "AN ORDINANCE": "Introduction",
            "A RESOLUTION": "Introduction",
            "Introduced": "Introduction",
            "ORDINANCE": "Introduction",
            "RESOLUTION": "Introduction",
            "WHEREAS": "Preamble",
            "NOW, THEREFORE": "Operative",
            "BE IT ORDAINED": "Operative",
            "ENACTED:": "Signature",
            "ADOPTED:": "Signature",
            "ATTESTED:": "Signature"
        }

    def process_file_to_db(self, doc_dir: str):
        final_layout_csv_path = os.path.join(doc_dir, "layout_final.csv")
        if not os.path.exists(final_layout_csv_path):
            print(f"File not found: {final_layout_csv_path}")
            return

        df = pd.read_csv(final_layout_csv_path, encoding='utf-8')
        filtered_df = df[~df['Layout'].str.startswith("Table")]
        text = filtered_df['Text'].to_list()
        hierarchical_chunks = self._process_hierarchical_chunk(text)
        print(f"Hierarchical chunks: {len(hierarchical_chunks)}")
        try:
            with self.client.batch.dynamic() as batch:
                for hierarchical_chunk in hierarchical_chunks:
                    first_line = hierarchical_chunk.split('\n')[0].strip().lstrip("'")
                    section_name = next((cat for key, cat in self.section_categories.items() if first_line.startswith(key)), "Uncategorized")
                    chunks = self._split_text(hierarchical_chunk)

                    for index, chunk in enumerate(chunks):
                        embedding = self.model_manager.get_embedding(chunk)
                        data_object = {
                            "source": os.path.basename(doc_dir),
                            "category": section_name,
                            "chunk_index": index + 1,
                            "text": chunk
                        }
                        legal_docs = self.client.collections.get("BAAI")
                        legal_docs.data.insert(properties=data_object, vector=embedding)
                        print(f"Added {os.path.basename(doc_dir)} ({section_name}) chunk {index + 1}")
        finally:
            self.client.close()

    def _process_hierarchical_chunk(self, text):
        sections, current_section, used_sections = [], [], set()
        first_whereas = True
        exclude_patterns = ["ORDINANCE NO.", "RESOLUTION NO.", "Regular Session"]
        has_nowtherefore = any("NOW, THEREFORE" in line for line in text)
        has_enacted = any("ENACTED:" in line or "ADOPTED:" in line for line in text)

        keywords = ["AN ORDINANCE", "A RESOLUTION", "WHEREAS", "NOW, THEREFORE", "ENACTED:", "ADOPTED:"]
        if not has_enacted:
            keywords = [kw if kw != "ENACTED:" else "ATTESTED:" for kw in keywords]
        if not has_nowtherefore:
            keywords = [kw if kw != "NOW, THEREFORE" else "BE IT ORDAINED" for kw in keywords]

        for line in text:
            line = line.strip()
            if any(p in line for p in exclude_patterns) and len(re.findall(r"[A-Za-z]", line.replace(",", ""))) < 50:
                continue

            matched_keyword = next((kw for kw in keywords if kw in line), None)

            if matched_keyword:
                if matched_keyword == "WHEREAS" and first_whereas:
                    first_whereas = False
                    if current_section:
                        sections.append("\n".join(current_section))
                    current_section = [line]
                    used_sections.add("WHEREAS")
                elif matched_keyword not in used_sections:
                    if current_section:
                        sections.append("\n".join(current_section))
                    current_section = [line]
                    used_sections.add(matched_keyword)
                else:
                    current_section.append(line)
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections

    def _split_text(self, text: str, chunk_size: int = 512):
        sentences = text.split("\n")
        chunks, current_chunk, current_size = [], [], 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_size = len(self.model_manager.tokenizer.tokenize(sentence))
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_size = [sentence], sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks




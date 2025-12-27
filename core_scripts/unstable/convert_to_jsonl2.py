#!/usr/bin/env python3
"""
Convert MTS-Dialog CSV files to JSONL conversation format.

Each row becomes a conversation with:
- System prompt (generic summarization instruction; NO section_header mention)
- User input (dialogue only)
- Assistant output (section_text as ground truth)

Two modes:
- --finetune: simple (no one-shot; short prompt)
- default: one-shot inference style (a single in-context example, still no section_header)
"""

import csv
import json
import argparse
from pathlib import Path


GENERIC_SYSTEM_PROMPT = (
    "You convert medical dialogues into concise, factual clinical notes.\n"
    "- Use ONLY information explicitly stated in the dialogue.\n"
    "- Do NOT invent details or rely on outside context.\n"
    "- Write in clear prose (no bullet points unless the dialogue clearly supports lists).\n"
    "- Keep neutral clinical tone and standard grammar.\n"
    "- If something is not stated, omit it."
)

ONE_SHOT_USER_PREFIX = "Summarize the following medical dialogue into a clinical note:\n\n"
FINETUNE_USER_PREFIX = ONE_SHOT_USER_PREFIX  # same content; shorter message stack


def convert_csv_to_jsonl(input_file, output_file, one_shot_index=42, alternate_index=100, finetune=False):
    """
    Convert CSV file to JSONL format with conversational structure.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output JSONL file
        one_shot_index: Index of the example to use as one-shot (default: 42)
        alternate_index: Alternative index to use when processing the one_shot_index row (default: 100)
        finetune: If True, use simplified format for fine-tuning (no one-shot, shorter prompt)
    """
    conversations = []

    # Load all data
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        raise ValueError(f"No rows found in {input_file}")

    # -------- Fine-tuning format (no one-shot) --------
    if finetune:
        for row in all_rows:
            dialogue = (row.get('dialogue') or '').strip()
            section_text = (row.get('section_text') or '').strip()

            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": GENERIC_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": FINETUNE_USER_PREFIX + dialogue
                    },
                    {
                        "role": "assistant",
                        "content": section_text
                    }
                ],
                "metadata": {
                    "id": row.get('ID', ''),
                    "section_header": row.get('section_header', ''),  # kept for analysis only
                }
            }
            conversations.append(conversation)

    # -------- One-shot inference format (ICL) --------
    else:
        # Defensive bounds for example indices
        if len(all_rows) <= one_shot_index:
            one_shot_index = max(0, min(42, len(all_rows) - 1))
        if len(all_rows) <= alternate_index:
            alternate_index = max(0, min(100, len(all_rows) - 1))

        one_shot_example = all_rows[one_shot_index]
        alternate_example = all_rows[alternate_index]

        one_shot_dialogue = (one_shot_example.get('dialogue') or '').strip()
        one_shot_answer = (one_shot_example.get('section_text') or '').strip()

        alternate_dialogue = (alternate_example.get('dialogue') or '').strip()
        alternate_answer = (alternate_example.get('section_text') or '').strip()

        for idx, row in enumerate(all_rows):
            # Use an alternate example when we hit the main one-shot row
            ex_dialogue = alternate_dialogue if idx == one_shot_index else one_shot_dialogue
            ex_answer = alternate_answer if idx == one_shot_index else one_shot_answer

            row_dialogue = (row.get('dialogue') or '').strip()
            row_answer = (row.get('section_text') or '').strip()

            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": GENERIC_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": ONE_SHOT_USER_PREFIX + ex_dialogue
                    },
                    {
                        "role": "assistant",
                        "content": ex_answer
                    },
                    {
                        "role": "user",
                        "content": ONE_SHOT_USER_PREFIX + row_dialogue
                    },
                    {
                        "role": "assistant",
                        "content": row_answer
                    }
                ],
                "metadata": {
                    "id": row.get('ID', ''),
                    # Keeping for bookkeeping only; NOT surfaced in prompts:
                    "section_header": row.get('section_header', ''),
                    "one_shot_example_id": (alternate_example if idx == one_shot_index else one_shot_example).get('ID', '')
                }
            }
            conversations.append(conversation)

    # Write JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    return len(conversations)


def main():
    parser = argparse.ArgumentParser(description='Convert MTS-Dialog CSV files to JSONL format')
    parser.add_argument('--input-dir', type=str, default='./data',
                        help='Input directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='./data/jsonl',
                        help='Output directory for JSONL files')
    parser.add_argument('--files', nargs='*',
                        help='Specific files to convert (if not specified, converts all CSV files)')
    parser.add_argument('--one-shot-index', type=int, default=42,
                        help='Index of the example to use as one-shot (default: 42)')
    parser.add_argument('--alternate-index', type=int, default=100,
                        help='Alternative index when processing the one-shot example itself (default: 100)')
    parser.add_argument('--finetune', action='store_true',
                        help='Use simplified format for fine-tuning (no one-shot, shorter prompt)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of files to process
    if args.files:
        csv_files = [input_dir / f for f in args.files if f.endswith('.csv')]
    else:
        csv_files = list(input_dir.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    # Process each file
    for csv_file in csv_files:
        # Add suffix for finetune files
        if args.finetune:
            output_file = output_dir / f"{csv_file.stem}-finetune.jsonl"
        else:
            output_file = output_dir / f"{csv_file.stem}.jsonl"

        print(f"Converting {csv_file.name}...")
        try:
            count = convert_csv_to_jsonl(
                csv_file,
                output_file,
                one_shot_index=args.one_shot_index,
                alternate_index=args.alternate_index,
                finetune=args.finetune
            )
            print(f"  ✓ Converted {count} conversations to {output_file}")
            if args.finetune:
                print(f"    Using fine-tune format (no one-shot examples, concise prompt; no section_header in prompts)")
            else:
                print(f"    Using example {args.one_shot_index} as one-shot (alternate: {args.alternate_index}); no section_header in prompts")
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")

    print(f"\nConversion complete! JSONL files saved in {output_dir}")


if __name__ == "__main__":
    main()

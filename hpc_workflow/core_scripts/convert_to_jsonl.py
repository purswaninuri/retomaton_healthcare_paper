#!/usr/bin/env python3
"""
Convert MTS-Dialog CSV files to JSONL conversation format.
Each row becomes a conversation with:
- System prompt (using section_header)
- User input (dialogue)
- Assistant output (section_text)
"""

import csv
import json
import argparse
from pathlib import Path


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
    all_rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if finetune:
        # Simplified format for fine-tuning
        for row in all_rows:
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a concise medical scribe, summarize the notes provided in the specified format."
                    },
                    {
                        "role": "user",
                        "content": row['context']
                    },
                    {
                        "role": "assistant",
                        "content": row['impression']
                    }
                ]
            }
            conversations.append(conversation)
    else:
        # One-shot format for inference
        # Get the one-shot example
        if len(all_rows) <= one_shot_index:
            one_shot_index = min(42, len(all_rows) - 1)
        if len(all_rows) <= alternate_index:
            alternate_index = min(100, len(all_rows) - 1)

        one_shot_example = all_rows[one_shot_index]
        alternate_example = all_rows[alternate_index]

        # Create conversations with one-shot examples
        for idx, row in enumerate(all_rows):
            # Use alternate example when processing the row that's used as the main one-shot
            if idx == one_shot_index:
                example_to_use = alternate_example
            else:
                example_to_use = one_shot_example

            # Create conversation structure with one-shot example
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise medical scribe, summarize the notes provided in the specified format."
                    },
                    {
                        "role": "user",
                        "content": "Summarise the following long note: {}\n".format(example_to_use['context'])
                    },
                    {
                        "role": "assistant",
                        "content": example_to_use['impression']
                    },
                    {
                        "role": "user",
                        "content": "Summarise the following long note: {}\n".format(row['context'])
                    },
                    {
                        "role": "assistant",
                        "content": row['impression']
                    }
                ],
                
            }
            conversations.append(conversation)

    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    return len(conversations)


def main():
    parser = argparse.ArgumentParser(description='Convert MIMIC Radio CSV files to JSONL format')
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
                print(f"    Using fine-tune format (no one-shot examples, concise prompt)")
            else:
                print(f"    Using example {args.one_shot_index} as one-shot (alternate: {args.alternate_index})")
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")
    
    print(f"\nConversion complete! JSONL files saved in {output_dir}")


if __name__ == "__main__":
    main()
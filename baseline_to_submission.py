import sys
import zipfile

def convert_to_submission(input_path, output_path, zip_path):
    print("Reading " + input_path)
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
    print(f"Formatting {len(lines)} lines to comma-separated values...")
    result_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            result_lines.append(f"{parts[0]}, {parts[1]}\n")
            
    print("Writing " + output_path)
    with open(output_path, 'w') as f:
        f.writelines(result_lines)
        
    print("Zipping to " + zip_path)
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.write(output_path, arcname='result.txt')
        
    print(f"Done! Please upload {zip_path} to CodaBench.")

if __name__ == '__main__':
    convert_to_submission('baseline/test_predict_output.txt', 'result.txt', 'baseline_submission.zip')

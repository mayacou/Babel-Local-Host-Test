import random
import json

sample_size = 5

def get_total_lines(filename):
   # Returns the total number of lines in a file without loading it into memory.
   with open(filename, "r", encoding="utf-8") as file:
      return sum(1 for _ in file)

def sample_consecutive_lines(source_file, reference_file):
   """
   Reads two large files in a streaming manner, selects sample_size consecutive lines from
   a random starting point, and returns them as a list of dicts with "input" and "reference".
   """
   # Get total number of lines
   total_lines = get_total_lines(source_file)

   # Pick a random start line between 0 and (total_lines - sample_size)
   start_index = random.randint(0, total_lines - sample_size)
   
   sampled_pairs = []
   # Open both files once; skip lines up to start_index, then read sample_size lines
   with open(source_file, "r", encoding="utf-8") as src, open(reference_file, "r", encoding="utf-8") as ref:
      # Skip lines until we reach start_index
      for _ in range(start_index):
         next(src, None)
         next(ref, None)
        
      # Read the next 'sample_size' lines and add to sampled_pairs dictionary
      for _ in range(sample_size):
         src_line = next(src).rstrip("\n")
         ref_line = next(ref).rstrip("\n")
            
         sampled_pairs.append({
            "input": src_line, 
            "reference": ref_line
         })
         
   return sampled_pairs

def save_to_json(data, output_json):
   with open(output_json, "w", encoding="utf-8") as json_file:
      json.dump(data, json_file, ensure_ascii=False, indent=4)
    
   print(f"✅ JSON data saved to {output_json}")

if __name__ == "__main__":
   # CHANGE FILE NAMES HERE
   source_file = "Europarl.en-fr.en"
   reference_file = "Europarl.en-fr.fr"
   
   output_json = "data.json"

   # Sample sample_size consecutive lines and then save them to JSON
   sampled_data = sample_consecutive_lines(source_file, reference_file)
   save_to_json(sampled_data, output_json)

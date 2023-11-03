#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openai
import csv
from tqdm import tqdm

# Set up OpenAI API credentials
openai.api_key = "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"

# Define the prompt
prompt = "Create a review without starting the sentence with a pronoun based on the following example:\n"

# Read the CSV file
with open(r"C:\Users\Ryan\Desktop\Python Projects\generated_reviews_extra.csv", "r") as input_file:
    with open(r"C:\Users\Ryan\Desktop\Python Projects\generated_reviews_final_extra3.csv", "w", newline='') as output_file:
        writer = csv.writer(output_file)
        
        # Write the header row
        writer.writerow(["category", "rating", "label", "generated_review"])
    
        reader = csv.DictReader(input_file)
        # Loop through each row of the file and generate reviews
        for i, row in tqdm(enumerate(reader), total=20216):
            # Stop after generating specific number of reviews
            if i >= 20216:
                break
            
            # Get the text from the current row
            example_text = row["text_"]
            
            # Add the example text to the prompt
            full_prompt = prompt + example_text
            
            # Get the response from GPT-3
            response = openai.Completion.create(
                engine='text-davinci-003',  # Determines the quality, speed, and cost.
                frequency_penalty=0.5,         # stops the language from being to fragmented 
                temperature=0.75,            # Level of creativity in the response
                prompt=full_prompt,              # What the user typed in
                max_tokens=40,            # Maximum tokens in the prompt AND response
                n=1,                        # The number of completions to generate
                stop=None,  
            )
            
            # Get the generated review from the response
            generated_review = response.choices[0].text.strip()
            
            # Check if the generated review has more than 3 words
            if len(generated_review.split()) > 3:
                # Write the generated review to the output file
                writer.writerow([row["category"], row["rating"], "CG", generated_review])


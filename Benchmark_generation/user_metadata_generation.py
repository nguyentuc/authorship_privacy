import openai
import os
# Set up the API key
openai.api_key = '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'

json_template = """
{
    "title": "General Profile Dataset",
    "description": "This dataset contains professional and personal information, including employment history, skills, credentials, followers, and activity highlights for various individuals.",
    "columns": {
      "name": {
        "description": "Full name of the individual.",
        "data_type": "String"
      },
      "profession": {
        "description": "A short description of the individual's professional or personal standing.",
        "data_type": "String"
      },
      "profile_image": {
        "description": "URL of the individual's profile image.",
        "data_type": "String (URL)"
      },
      "followers": {
        "description": "Number of followers the individual has on social or professional platforms.",
        "data_type": "String (likely integer)"
      },
      "following": {
        "description": "Number of people the individual is following.",
        "data_type": "String (likely integer)"
      },
      "credentials": {
        "description": "A list of credentials and background information about the individual, including education, personal interests, and professional roles.",
        "data_type": "Array of strings"
      },
      "highlights": {
        "description": "Key highlights or accomplishments from the individual's profile, including education, content views, and activity on various platforms.",
        "data_type": "Array of strings"
      },
      "location": {
        "description": "Location of the individual, including city, state, and country.",
        "data_type": "String"
      },
      "skills": {
        "description": "List of skills the individual possesses, with optional endorsement counts.",
        "data_type": "Array of objects",
        "structure": {
          "name": "String (skill name)",
          "endorsements": "Integer (number of endorsements)"
        }
      },
      "employment_history": {
        "description": "List of jobs or positions the individual has held, including title, company, and duration.",
        "data_type": "Array of objects",
        "structure": {
          "from_date": "String (start date)",
          "to_date": "String (end date or 'None' if still employed)",
          "duration": "String (length of employment)",
          "title": "String (job title)",
          "company": "String (employer)",
          "location": "String (job location)",
          "emp_type": "String (employment type)"
        }
      },
      "certifications": {
        "description": "List of certifications or educational achievements, with details of the issuer and issue date.",
        "data_type": "Array of objects",
        "structure": {
          "title": "String (certification title)",
          "issuer": "String (organization issuing the certification)",
          "date_issued": "String (date issued)",
          "cert_id": "String (optional certification ID)",
          "cert_link": "String (optional certification link)"
        }
      },
      "content_views": {
        "description": "Total number of content views across platforms (e.g., Quora).",
        "data_type": "String (likely integer)"
      },
      "spaces_active": {
        "description": "Number of spaces or groups the individual is actively participating in.",
        "data_type": "Integer"
      },
      "join_date": {
        "description": "Date when the individual joined a platform.",
        "data_type": "String (date format)"
      },
      "email": {
        "description": "The contact email address of the individual.",
        "data_type": "String"
      },
      "profile_url": {
        "description": "URL of the individual's profile on a professional or social platform (e.g., LinkedIn, Quora).",
        "data_type": "String (URL)"
      },
      "additional_information": {
        "description": "Any additional relevant information about the individual's profile.",
        "data_type": "String (optional)"
      }
    },
    "license": "No specific license information provided.",
    "date_created": "Not specified",
    "date_updated": "Not specified",
    "size": "Metadata size unspecified, varies with profile",
    "source": "Various social media or professional platforms, like LinkedIn or Quora."
  }
"""

# Function to ask ChatGPT to synthesize a user profile
def formatting_writer_profile(json_template, sentence):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      response_format={ "type": "json_object" },
      seed=42,
      messages=[
       {"role": "system", "content": "Job: Folow json template of user, format the input text with the json template. The template is: {json_template}. The input is list of text that need to format is:"},
        {"role": "user", "content": sentence}]
    )
    return response.choices[0].message.content

# reading all the txt file
folder_path ='/media/volume/arkai-lab-data-private/Coding/AA/Synthetic_generation/linkedin_userprofile/quora/'
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as f:
        content = f.read()
        print(type(content))
        exit()
        # Generate the synthesized profile
        profile = formatting_writer_profile(json_template, content)
        print("Formated Profile:\n", profile)

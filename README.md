2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the Poetry environment:
   ```bash
   poetry env use python3.12
   ```

4. Create a `.env` file in the project root directory with your API keys:
   ```
   LLMWHISPERER_API_KEY=your_llmwhisperer_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key  

   ```

## Usage


Process surveys using Gemini 2.5 preview via OpenRouter:
```bash
python main.py path/to/surveys
```


## Output

All parsed-surveys are saved to the `output` directory with the following structure:

- `*.json`: Main output JSON files in as provided in the Schema (original_file_name.json)
- `ocr/*.txt`: Raw OCR text extracted from each PDF
- `prompts/*.txt`: Prompts used for LLM processing (when debugging is enabled)
- `temp/`: Temporary files (cleaned up after processing)

## Required API Keys

1. **LLMWhisperer API Key (Required)**
   - Used for OCR processing to extract text from PDFs
   - Obtain from [LLMWhisperer](https://llmwhisperer.io)


4. **OpenRouter API Key (Optional)**
   - Required for using Gemini 2.5 preview
   - Obtain from [OpenRouter](https://openrouter.ai)




## Workflow:


# Survey Parser: PDF & Document Survey Extraction System





### Processing Pipeline

#### 1. File Discovery and Preparation
- Scans input directory for files matching specified patterns(The current workflow only supports .pdf, .doc and .docx )
- Creates necessary output directories
- Cleans previous output if present

#### 2. Batch Processing with Load Balancing
- Implements provider-specific concurrency limits
- Processes multiple files in parallel with controlled rate limiting

#### 3. OCR
- OCR API only takes in one page at a time using LLMWhisperer API
- LLMWhisperer takes in as many file pages as possible 
before hitting rate limit (main bottleneck but can be fixed with on premise soltution as well as increased rate limits)

#### 4.  Text Chunking
- Created overlapping chunks on OCRed pages that may span page boundaries for survey 
- Includes context from previous and next pages
- Marks page boundaries with clear delimiters
- Handles questions and answers that span across pages: ie, 
scenarios where question are asked in on page and answer are found on the next.

#### 5. LLM Processing using gemini 2.5:
- LLM processes OCRed text chunks using a prompt such as:
```
    You are an expert at analyzing survey data from OCR-processed documents.

    Your task is to extract all questions and answers from the following survey text chunk in a structured format.
    
    This is a chunk of text (chunk_1) that may span across page boundaries.
    The chunk contains content from multiple pages with '<<<' markers indicating page breaks.
    

    IMPORTANT CHUNK PROCESSING INSTRUCTIONS:
    - This chunk may contain content from multiple pages
    - The '<<<' markers indicate page breaks
    - Pay special attention to questions and answers that span across these page break markers
    - When you see a question right before a '<<<' marker, look for its answer right after the marker
    - When you see an answer at the beginning of the chunk (before any question), it likely belongs to a question from the previous page

    CRITICAL SPATIAL AWARENESS INSTRUCTIONS:
    - Survey forms often have complex layouts where answers don't always appear directly below or beside their questions
    - Scan the ENTIRE document space including headers, footers, margins, and floating elements for potential answers
    - Pay special attention to:
      * Headers/letterheads (often contain institution/company names, dates, ID numbers)
      * Floating text boxes/fields (may contain answers to questions that appear elsewhere)
      * Text in different columns that may be related
      * Information in margins or page corners
      * Titles and subtitles that contain relevant information
    - Be aware that some answers may appear visually separated from their questions in the original form
    - For questions asking about names, addresses, dates, or identifiers, check page headers and document metadata
    - Match answers to questions based on context and content, not just proximity
    - When dealing with page breaks ('<<<'), remember that important contextual information might be split across pages

    For each question:
    1. Extract the question ID (e.g., Q2.3, Q3.1) if present. If no ID is explicitly shown, generate a sequential one.
    2. Extract the complete question text, including any asterisks (*) that denote required fields.
    3. Identify all possible answer options.
    4. Determine which answer was selected based on:
       - Checkboxes: Look for [X], ☑, or similar marks
       - Radio buttons: Look for filled circles or selected options
       - Text fields: Extract the text entered
       - Multiple selections: If multiple options are checked, include ALL of them as a list
    5. Identify any section headers or organizational elements.

    Very important rules:
    - Pay special attention to cases where a question appears right before a '<<<' marker and its answer appears right after it
    - Look for standalone answers at the top of a chunk that might belong to questions from the previous page
    - For isolated answers like "2+" or "Yes" that appear at the start of the chunk, try to match them to the last question that appears in the previous section (before the first '<<<' marker)
    - For questions at the end of the chunk (after the last '<<<' marker) that don't have answers, mark them as "[INCOMPLETE]"
    - If there are multiple checked boxes for a question, include all selected options as an array
    - For free-text responses, include the complete answer text

    Format your response in JSON structure as follows:
    ```json
    {
      "sections": [
        {
          "title": "Section Title",
          "questions": [
            {
              "id": "Q1.1",
              "text": "Full question text",
              "options": ["Option 1", "Option 2", "Option 3"],
              "selected": "Option 2"
            },
            {
              "id": "Q1.2",
              "text": "Multiple-select question text",
              "options": ["Option A", "Option B", "Option C", "Option D"],
              "selected": ["Option A", "Option C"]
            },
            ...
          ]
        },
        ...
      ]
    }
    ```

    If a question appears incomplete, extract what you can and indicate with "[INCOMPLETE]" in the text field.
    If something is ambiguous, indicate this with "[UNCLEAR]".

    Here is the OCR text chunk to analyze:

    

                                           Innovo Research 
                                      IBC Authorization Form 
                                 For IBC Review of Clinical Trials 

The following documentation establishes Advarra as an entity externally administering the 
Institutional Biosafety Committee (IBC) for: 

                                                                Type of Organization (check all 
  Site Name:         Starling Physicians, P.C.                             that apply): 
 Site Contact                                                                           Government 
 Name:               Chantal Pelzer                           Academic       [ ]        [ ] 
 Site Contact        Site Feasibility and Activation                                    Commercial 
  Title:             Manager                                  Research        [ ]       [ ] 
 Site Contact                                                                           pharmaceutical) (e.g. biotech, 
 Email:              Start-up@innovoresearch.com             Institute        [ ] 
 Site Contact 
 Phone:              336-972-2909                            Hospital/Clinic [X]  
                     300 Kensington Ave, New Britain, 
 Site Address:       CT 06051                                Other [ ]   Other 
  Has your site ever participated in research supported by the NIH?                 Yes [X]     No [ ] 

Advarra has agreed to serve as an IBC of record for this site. Advarra will fully comply with the 
requirements set forth in NIH Guidelines for Research Involving Recombinant or Synthetic 
Nucleic Acid Molecules (NIH Guidelines). The IBC administered by Advarra has knowledge of the 
local site characteristics including Investigator training, laboratory conditions and operating 
procedures. Advarra's IBC review is independent of Advarra's IRB review services. As such, any 
research involving human subjects must be reviewed and approved by the IRB prior to initiation. 

Signature of IBC Representative for Advarra: 

Daniel Eisenman, PhD, RBP, SM (NRCM), CBSP                           Date 
Executive Director of Biosafety Services 
Advarra, Inc. 

This site accepts the authority of Advarra as the externally administered IBC for studies 
conducted at this site. I have the authority and responsibility to implement the IBC's directives 
at the site above. 

Authorized Site Representative: 

 Chantal Pelzer                                       Site Feasibility and Activation Manager 
 Name                                                 Title 

 Chantal Pelzer                                       November 11, 2024 
 Signature                                            Date 

G:\Shared drives\/BC\IBC Drive\/AFs\IAF - New 2023| Advarra NIH Registration Form 2023.docx 
<<<

                                                                                        THERAPY 

                                                                                     GENE        READY 
 ADVARRA 
   advancing better research                                                             ADVARRA® 

                          Gene Therapy Ready™" Onboarding Survey 

Contact info for sponsor/CRO inquiries: 
Site Name:         Starling Physicians, P.C.        Site Contact      Site Feasibility and 
                                                    Title:            Activation Manager 
Site Address:      300 Kensington Ave, New          Site Contact      Start- 
                   Britain, CT 06051                Email:            up@innovoresearch.com 
Site Contact       Chantal Pelzer                   Site Contact      336-972-2909 
Name:                                               Phone: 

Please check all that apply: 
    [X]  This site has conducted clinical trials at this location. 

    [ ] This site has received IBC approval for previous research. 
    [ ] This site is currently conducting or has previously conducted research involving 
       genetically engineered materials. 
    [ ] This site anticipates conducting research involving genetically engineered materials in: 
            [ ] 1 month      [ ] 3 months                      [ ] 6 months     [ ] 9 months 
                      [X]  12+ months 
    [X]  This site would like to complete a facility site inspection before study is initiated. 

                             Anticipated Investigational Products 
    [X] Genetically modified vaccines (e.g. mRNA, plasmids, viral-based) [ ] Gene therapy 
    [ ] Genetically engineered cell therapy                      [ ] Other Click or tap here to 
   enter text. 

                                 Therapeutic Areas of Interest 
   [X] Bacterial and Fungal Diseases             V [X] Muscle, Bone, and Cartilage Diseases 
   [X] Behaviors and Mental Disorders            [X]  Nervous System Diseases 
   [X] Blood and Lymph Conditions                [X] Nutritional and Metabolic Diseases 
   [X] Cancers and Other Neoplasms               [X]  Parasitic Diseases 
   [X] Digestive System Diseases                 [X] Rare Diseases 
   V [X] Diseases and Abnormalities at or        [X] 
   before Birth                                  V   Regenerative Medicine 

   V [X] Ear, Nose, and Throat Diseases          [X]  Diseases Respiratory Tract (Lung and Bronchial) 

   V [X] Eye Diseases                            [X] Skin and Connective Tissue Diseases 

   Diseases V [X] Gland and Hormone Related      [ ] Substance Related Disorders 

   [X]  Heart and Blood Diseases                Conditions [X] Urinary Tract, Sexual Organs, and Pregnancy 

   [X] Immune System Diseases                    [X]  Viral Diseases 
   V [X] Mouth and Tooth Diseases                [X] Wounds and Injuries 
<<<

                                                                                             THERAPY Y 

                                                               OtherClick or tap here to ENE enter     READY 
                                                    [ ] 
ADVARRA                                                 Other text. 
   advancing better research 
                                                                                              ADVARRA® 
<<<
```
The OCRed section in the prompt above: 
![OCR Survey Form Example](images/image%20copy.png)
Keeping OCR text exactly in its natural format like when feeding into LLM is crucial for keeping the system robust in handling surveys in a wide variety of of formats. The OCRed texts long spaces and emtpy lines maintain true form and human readability which is crucial as some of these surveys have complex layouts.


#### 6.  Data Merging
- Combines processes LLM data from multiple chunks and pages
- Handles duplicate questions appropriately
- Resolves conflicts in selected answers


#### 7. Provider Classification
- Identifies the survey provider (SurveyMonkey, Google Forms, Qualtrics, etc.)
- Uses full document content for accurate identification
- Employs a page-based voting mechanism using Gemini where each page's classification contributes to the final :
- Detects provider-specific indicators like URLs and name in text

```

You're evaluating a survey document to identify which platform or software created it.

TASK: Analyze the OCR text from this document and identify the survey provider.

ALLOWED RESPONSES - Choose exactly ONE of these:
- SURVEY_MONKEY
- GOOGLE_FORMS
- QUALTRICS
- PDF
- MICROSOFT_WORD
- MICROSOFT_FORMS
- NULL

KEY INDICATORS TO LOOK FOR:
- URLs containing domains like "surveymonkey.com", "forms.office.com", "forms.microsoft.com", "docs.google.com/forms", or "qualtrics.com"
- Copyright notices or footers mentioning any of these providers
- Provider-specific layouts, question formatting, or styling patterns, provider name in the document
- Branding elements or logos from these platforms

IMPORTANT DISTINCTIONS:
- Microsoft Forms will specifically have a forms.office.com URL or Microsoft Forms branding
- Look for file paths with .doc or .docx extensions which indicate Microsoft Word documents

DOCUMENT DETAILS:
This is a .DOCX document.

DOCUMENT OCR TEXT:


                                           Innovo Research 
                                      IBC Authorization Form 
                                 For IBC Review of Clinical Trials 

The following documentation establishes Advarra as an entity externally administering the 
Institutional Biosafety Committee (IBC) for: 

                                                                Type of Organization (check all 
  Site Name:         Starling Physicians, P.C.                             that apply): 
 Site Contact                                                                           Government 
 Name:               Chantal Pelzer                           Academic       [ ]        [ ] 
 Site Contact        Site Feasibility and Activation                                    Commercial 
  Title:             Manager                                  Research        [ ]       [ ] 
 Site Contact                                                                           pharmaceutical) (e.g. biotech, 
 Email:              Start-up@innovoresearch.com             Institute        [ ] 
 Site Contact 
 Phone:              336-972-2909                            Hospital/Clinic [X]  
                     300 Kensington Ave, New Britain, 
 Site Address:       CT 06051                                Other [ ]   Other 
  Has your site ever participated in research supported by the NIH?                 Yes [X]     No [ ] 

Advarra has agreed to serve as an IBC of record for this site. Advarra will fully comply with the 
requirements set forth in NIH Guidelines for Research Involving Recombinant or Synthetic 
Nucleic Acid Molecules (NIH Guidelines). The IBC administered by Advarra has knowledge of the 
local site characteristics including Investigator training, laboratory conditions and operating 
procedures. Advarra's IBC review is independent of Advarra's IRB review services. As such, any 
research involving human subjects must be reviewed and approved by the IRB prior to initiation. 

Signature of IBC Representative for Advarra: 

Daniel Eisenman, PhD, RBP, SM (NRCM), CBSP                           Date 
Executive Director of Biosafety Services 
Advarra, Inc. 

This site accepts the authority of Advarra as the externally administered IBC for studies 
conducted at this site. I have the authority and responsibility to implement the IBC's directives 
at the site above. 

Authorized Site Representative: 

 Chantal Pelzer                                       Site Feasibility and Activation Manager 
 Name                                                 Title 

 Chantal Pelzer                                       November 11, 2024 
 Signature                                            Date 

G:\Shared drives\/BC\IBC Drive\/AFs\IAF - New 2023| Advarra NIH Registration Form 2023.docx 
<<<
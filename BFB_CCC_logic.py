import pandas as pd
from PIL import Image
import spacy
import easyocr
import numpy as np
import pandas as pd
from skimage import io
from skimage.color import rgb2lab, deltaE_cie76
from sklearn.cluster import KMeans
from io import BytesIO
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
reader = easyocr.Reader(['en'])

w_RT = 0.5  # Adjust based on importance
w_RIT = 0.5  # Adjust based on importance

L_scores=[]
R_scores = []
C_scores=[]
Total_scores=[]

essential_attributes = ['Product Name', 'Brand Name', 'Price', 'Quantity/Size', 'Shade/Color', 'Ingredients', 'Product Type', 'Usage Instructions', 'Expiration Date', 'Manufacturing Date', 'Country of Origin', 'Special Features', 'Certifications']
info_categories = ['Benefits', 'Skin Type Compatibility', 'Packaging', 'Return Policy', 'Disclaimer']
high_risk = [
    "Bithionol",
    "Chlorofluorocarbon propellants",
    "Chloroform",
    "Halogenated salicylanilides",
    "Hexachlorophene",
    "Mercury compounds",
    "Methylene chloride",
    "Vinyl chloride",
    "Aluminum Zirconium Chlorohydrates"
]

moderate_risk = [
    "Oxybenzone",
    "Methylisothiazolinone",
    "Sodium Lauryl Sulphate",
    "Ethyl Acrylate",
    "Ethyl Methacrylate",
    "Methyl Methacrylate"
    "Fragrances",
    "Formaldehyde",
    "Toluene"
    "Phthalates",
    "Triclosan"
]

supported_labels=["organic", "natural", "hypoallergenic", "fragrance-free", "vegan", "no animal testing", "dermatologist-tested", "cruelty free", "environmentally-friendly"]

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

#----------------------------------------------Completeness Score----------------------------------------------

def calculate_LA(row):
    # Check if each cell is not NA for all essential attributes and sum the True values
    provided_attributes = sum(pd.notna(row[attribute]) for attribute in essential_attributes)
    total_attributes = len(essential_attributes)
    LA = provided_attributes / total_attributes
    return LA

def calculate_LD(row):
    # Similarly, check if each cell is not NA for all information categories and sum the True values
    provided_info = sum(pd.notna(row[category]) for category in info_categories)
    total_categories = len(info_categories)
    LD = provided_info / total_categories
    return LD

#----------------------------------------------Correctness Score----------------------------------------------

def basic_image_quality_assessment(image_path):
    try:
        img = Image.open(image_path)

        width, height = img.size
        if width > 1024 and height > 768:
            return 1.0
        else:
            return 0.5
    except Exception as e:
        print(f"Error assessing image quality for {image_path}: {e}")
        return 0.0

def calculate_RT(row, wing=0.5, wsize=0.5, wpname=0.5, wbname=0.5 ):

    image_path = row['Image Path']  # Assuming this is a valid local path

    ingredients_listed = str(row['Ingredients']) if pd.notna(row['Ingredients']) else ""
    quantity_listed = str(row['Quantity/Size']) if pd.notna(row['Quantity/Size']) else ""
    name = str(row['Product Name']) if pd.notna(row['Product Name']) else ""
    bname = str(row['Brand Name']) if pd.notna(row['Brand Name']) else ""
    try:
        result = reader.readtext(image_path, detail=0)  # Directly use the local path
        extracted_text = ' '.join(result).lower()

        I_acc = 1 if any(ingredient.lower() in extracted_text for ingredient in ingredients_listed.split(',')) else 0
        Q_acc = 1 if quantity_listed.lower() in extracted_text else 0
        P_acc = 1 if name.lower() in extracted_text else 0
        B_acc = 1 if bname.lower() in extracted_text else 0

        RT = wing * I_acc + wsize * Q_acc +wpname * P_acc + wbname * B_acc
        return RT

    except Exception as e:
        print(f"Error processing image or calculating RT for {image_path}: {e}")
        return 0

def color_name_to_lab(color_name):
    color_name_to_lab_map = {
        'midnight black': [0, 0, 0],  # Very dark colors tend to have low L values and minimal a/b
        'shade-4': [50, 0, -50],  # Placeholder value, "Shade-4" is not descriptive of color
        'black': [0, 0, 0],  # Similar to Midnight Black
        'light pink': [80, 20, 5],  # Light pink has a high L value, positive a, and low b
        'passion pink': [60, 40, 10],  # More saturated pink than Light Pink
        'beige': [70, 5, 20],  # Beige colors have moderate L and positive a and b
        'beige 03': [70, 5, 20],  # Assuming similar to Medium Beige
        'mauve':[78.531, 31.497, -32.483],
        'brown':[40.44, 27.498, 50.142],
        'nude':[78.829, 9.157, 22.456],
        'ivory':[99.64, -2.551, 7.163],
        'white': [100, 0, 0],
        'red':[53.241, 80.092, 67.203],
    }
    
    # Define a default LAB value for unknown or NA colors
    default_lab_value = [0, 0, 0]  # Adjust this as needed

    # Check if color_name is NA or not in the mapping
    if pd.isna(color_name) or color_name.lower() not in color_name_to_lab_map:
        print(f"Unknown or NA color name: '{color_name}'. Using default LAB values.")
        return default_lab_value
    else:
        return color_name_to_lab_map.get(color_name.lower(), default_lab_value)

def get_dominant_color(image_path, num_colors=1):
    """
    Extract the dominant color(s) from a local image.
    """
    img = io.imread(image_path)  # Load image from local path
    img_lab = rgb2lab(img.reshape((-1, 3)))  # Convert to LAB color space

    # Use KMeans clustering to find the most dominant color
    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=0).fit(img_lab)
    dominant_colors = kmeans.cluster_centers_

    return dominant_colors[0]  # Return the first dominant color

def calculate_RIT(row):
    image_path = row['Image Path']  # Assuming this is a valid local path
    product_description = str(row['Description']) if pd.notna(row['Description']) else ""
    shade_description = str(row['Shade/Color']) if pd.notna(row['Shade/Color']) else ""

    # Initialize the color consistency score to 0 to penalize missing 'Shade/Color'
    color_consistency_score = 0

    try:
        result = reader.readtext(image_path, detail=0)  # Directly use the local path
        extracted_text = ' '.join(result)
        keywords = product_description.split()
        text_matches = sum(keyword.lower() in extracted_text.lower() for keyword in keywords)
        text_consistency_score = text_matches / len(keywords) if keywords else 0

        # Proceed with color analysis only if 'Shade/Color' is provided
        if shade_description:
            dominant_color_lab = get_dominant_color(image_path)
            shade_lab = color_name_to_lab(shade_description)
            if shade_lab is not None:
                delta_e = deltaE_cie76(dominant_color_lab, shade_lab)
                color_consistency_score = 1 - (delta_e / 100)  # Adjust scoring as needed

        # Weighted sum of consistency scores, considering text and color
        wcolor = 0.5  # Adjust weights as needed
        wtext = 0.5  # Assuming equal weight for text and color for demonstration
        RIT = (wcolor * color_consistency_score + wtext * text_consistency_score)

        return RIT

    except Exception as e:
        print(f"Error processing image or calculating consistency for {image_path}: {e}")
        return 0  # Return 0 to penalize rows where an error occurs in processing
    
#----------------------------------------------Compliance Score----------------------------------------------

# Function to preprocess and stem the words in a sentence
def preprocess(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Function to check if a given ingredient is a variation of any high-risk or moderate-risk ingredient
def check_variation(ingredient, risk_list):
    processed_ingredient = preprocess(ingredient)
    for risky_ingredient in risk_list:
        processed_risky_ingredient = preprocess(risky_ingredient)
        if set(processed_ingredient).intersection(processed_risky_ingredient):
            return True
    return False

# Function to calculate Compliance Labeling Rating (CLR)
def calculate_CLR(ingredients, labels, supported_claims, weights):
    ingredient_compliance = 1
    label_compliance = 0

    for ingredient in ingredients:
        # Perform NLP-based analysis for ingredient risk assessment
        if check_variation(ingredient, high_risk):
            ingredient_compliance *= 0.25
        elif check_variation(ingredient, moderate_risk):
            ingredient_compliance *= 0.5

    for label in labels:
        # Perform NLP-based analysis for label validation
        if check_variation(label, supported_labels):
            label_compliance += 1


    CLR = weights['wing'] * ingredient_compliance + weights['wlab'] * label_compliance
    return CLR

#---------------------------------------------------------------------------------------------------------------

def calcCompleteness(data):
  
  for index, row in data.iterrows():
    LA = calculate_LA(row)
    LD = calculate_LD(row)
    LIQ = basic_image_quality_assessment(row['Image Path'])
    LA = 0 if LA is None else LA
    LD = 0 if LD is None else LD
    LIQ = 0 if LIQ is None else LIQ

    L = (LA + LD + LIQ) / 3
    L_scores.append(L * 30)

  return L_scores


#   data['Completeness Score'] = L_scores
# # Save the updated DataFrame to a new Excel file
#   updated_csv_file_path = 'updated_' + csv_file_path
#   data.to_csv(updated_csv_file_path, index=False)

def calcCorrectness(data):
  for index, row in data.iterrows():
    RT = calculate_RT(row)  # This function needs to be defined based on textual accuracy criteria
    RIT = calculate_RIT(row)  # Already defined to assess image-text consistency and color detection

    # Calculate the overall Correctness Score for each product
    R = w_RT * RT + w_RIT * RIT
    R_scores.append(R * 35)  # Scale to a value out of 35, as per your scoring system

  return R_scores
#   data['Correctness Score'] = R_scores
#   updated_csv_file_path = 'updated_' + csv_file_path
#   data.to_csv(updated_csv_file_path, index=False)

def calcCompliance(data):
  weights={'wing':18,'wlab':12}
  for index, row in data.iterrows():
    ingredients = str(row['Ingredients']) if pd.notna(row['Ingredients']) else ""
    ingredients= ingredients.split(',')
    labels = str(row['Certifications']) if pd.notna(row['Certifications']) else ""
    labels= labels.split(',')
    # Calculating CLR
    CLR = calculate_CLR(ingredients, labels, supported_labels, weights)
    C_scores.append(CLR)  # Scale to a value out of 35, as per your scoring system

  return C_scores
#   data['Compliance Score'] = C_scores
#   updated_csv_file_path = 'updated_' + csv_file_path
#   data.to_csv(updated_csv_file_path, index=False)

def main(data):
    # csv_file_path = 'product_dataset(1).csv'
    # data = pd.read_csv(csv_file_path, delimiter=',')
    correctness_scores = calcCorrectness(data)
    completeness_scores = calcCompleteness(data)
    compliance_scores = calcCompliance(data)

    scores = pd.DataFrame({'Correctness Score': correctness_scores, 'Completeness Score': completeness_scores, 'Compliance Score': compliance_scores})
    Total_scores = pd.DataFrame(columns = ['Scores'])
    total_score_list = []

    i = 0

    for index, row in scores.iterrows():

        corr = row['Correctness Score']
        compliance = row['Compliance Score']
        complete = row['Completeness Score']
        total=float(corr)+float(compliance)+float(complete)
        total_score_list = [total]
        # Scale to a value out of 35, as per your scoring system
        Total_scores.loc[len(Total_scores)] = total_score_list
    # data['Total Score'] = Total_scores
    return Total_scores
    # res_file_path = 'scores_' + res_file_path
    # data.to_csv(res_file_path, index=False)

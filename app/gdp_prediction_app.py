# Improt relevant Libraries

# Streamlit dependencies
import streamlit as st
import streamlit_survey as ss

# Data handling dependencies
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import joblib
import base64 # For background image
import os

# Data Analysis and Visualization
import plotly.express as px
import plotly.graph_objects as go


# Load the trained model
with open('trained_models/gb_model.pkl', 'rb') as trained_model:
    model = pickle.load(trained_model)

# Load the label encoder and scaler for preprocessing
label_encoder = joblib.load('trained_models/label_encoder.pkl')
scaler = joblib.load('trained_models/scaler.pkl')

# Load the test dataset
test_data = pd.read_csv('test_data.csv')

# Load dataset with actual 2019 GDPR values
data = pd.read_csv('SA_GDP_estimation_dataset.csv', usecols=['province', 'year', 'GDPR'])

# Defining the actual GDPR for 2019
data2 = data[data['year'] == 2019]
actual_gdp = data2['GDPR']


# Function to preprocess data and make predictions
def predict_gdp(province, year):
    # Filter the test data based on user inputs
    input_data = test_data[(test_data['province'] == province) & (test_data['year'] == year)]
    print("Input Data:")
    print(input_data)
    
    # Preprocess the input data    
    input_data['province'] = label_encoder.transform(input_data['province'])

    # Scale numerical features
    num_features = ['avg_sum_of_intensity', 'avg_mean', 'sum_of_percentage_light']
    input_data[num_features] = scaler.transform(input_data[num_features])

    # Make predictions using the trained model
    features = input_data
    predictions = model.predict(features)

    return predictions


# Set page title and favicon
st.set_page_config(
    page_title="GDP Prediction App",
    page_icon="📈",
    initial_sidebar_state="collapsed", # to hide side bar at start of app
)

# # Load additional configuration from a TOML file
# st.set_config_file('resources/config.toml')

# Home Page
def home_page():
    # Logo and title
    st.image('resources/team9_logo.png', use_column_width=280)
    st.write("_____")
    st.title('GDP Prediction App')

    #  # Increase font size and change font color for the title
    # st.markdown("<h1 style='color: #008080; font-size: 36px;'>GDP Prediction App</h1>", unsafe_allow_html=True)


    # App description
    st.write("""
    ##### 🚀 **Boost Your Insight into Economic Growth!** 📈

    ##### Unleash the power of cutting-edge technology to estimate GDP values with our GDP Prediction App. Powered by satellite nighttime images, this app provides you with rapid and accurate GDP predictions at your fingertips.
    """)

    # Dropdown for selecting the country
    st.subheader('Select Country')
    countries = ['South Africa', 'Nigeria', 'China']
    selected_country = st.selectbox('', countries)

    if selected_country == 'South Africa':
        # If South Africa is selected, show dropdown for provinces
        st.subheader('Select Province')        
        original_province_names = test_data['province'].unique()

        # Create a list of formatted names for display purposes
        display_province_names = [province.replace('_', ' ').title() for province in original_province_names]
        selected_province = st.selectbox('', display_province_names)       
    else:
        # If another country is selected, you can customize the UI accordingly
        st.warning("Province selection is currently available for only South Africa.")

    # Dropdown for selecting the year
    st.subheader('Select Year')
    selected_year = st.slider('', min_value=2019, max_value=2023, value=2019, step=1)

    # Button to trigger the prediction
    if st.button('Predict GDP'):

        # Retrieve the original province name based on the selected display name
        selected_province_index = display_province_names.index(selected_province)
        original_province_name = original_province_names[selected_province_index]

        # Make predictions
        predictions = predict_gdp(original_province_name, selected_year)

        # Display the predicted GDP as a whole number
        st.success(f'Predicted GDP for {selected_province}, {selected_year}: {int(predictions[0]):,} (Rands million)')
        

        # Display the model evaluation metric
        st.write("-----------")
        st.write('#### Model Evaluation Metric')

        # Calculate and display accuracy metrics if actual GDP values are available in test_data
        actual_gdp = data2.loc[(data2['province'] == original_province_name) & (data2['year'] == selected_year), 'GDPR'].values
 
        if actual_gdp.size > 0:
            # Calculate RMSE
            rmse = np.sqrt(np.mean((actual_gdp - predictions)**2))
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:,.2f} (Rands million)")

            st.write('#### Percentage Error')
            #To calculate % error - 1st calculate the mean of the observed(actual) values
            mean_observed = sum(actual_gdp) / len(actual_gdp)
            percentage_error = round((rmse / mean_observed) * 100 , 1) #Calculate the percentage error            
            st.write(f"This prediction is **{percentage_error}%** off the actual GDP Value")          
        else:
            st.warning("Actual GDP values not available for the selected province and year.")

        # st.metric(label="Predicted GDP", value=f"{int(predictions[0]):,} (Rands million)", delta=f"{rmse:,.2f}", delta_color="off")
   
 

# ---------------------------> DATA EXPLORATION PAGE <------------------------------

# Explore Data Page
def explore_data_page():
    # Page description
    st.write("""
    🌍 **Explore Satellite Images and Night Light Intensity Data** 📸

    Dive into the visual world of satellite images and explore the transformed night light intensity data for different provinces and months.
    """)

    # Dropdown for selecting the country
    st.subheader('Select Country')
    countries = ['South Africa', 'Nigeria', 'China']
    selected_country = st.selectbox('', countries)

    if selected_country == 'South Africa':
        # If South Africa is selected, show dropdown for provinces
        st.subheader('Select Province')        
        original_province_names = test_data['province'].unique()
        # Create a list of formatted names for display purposes
        display_province_names = [province.replace('_', ' ').title() for province in original_province_names]
        selected_province = st.selectbox('', display_province_names)
    else:
        # If another country is selected, you can customize the UI accordingly
        st.warning("Province selection is currently available for only South Africa.")

    # Dropdown for selecting the year
    st.subheader('Select Year')
    selected_year = st.slider('', min_value=2013, max_value=2023, value=2013, step=1)

    # Button to trigger the prediction
    if st.button('DISPLAY IMAGES'):

        # Retrieve the original province name based on the selected display name
        selected_province_index = display_province_names.index(selected_province)
        original_province_name = original_province_names[selected_province_index]

        image_path = 'resources/satellite_images/'
        image_files = os.listdir(image_path)
        image_files = [file for file in image_files if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.gif')]
        
        filtered_image_files = [file for file in image_files if original_province_name.lower() in file.lower()]

        if len(filtered_image_files) > 0:
            # Display the first image matching the selected province
            image_path = os.path.join(image_path, filtered_image_files[0])
            st.image(image_path, caption=f'{selected_year} Satellite images for {selected_province}', use_column_width=True)
        else:
            st.warning(f"No images found for the selected province: {selected_province}")


# -------------------------> Add background image Function <--------------------------
# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Loading markdown files for long text
def read_markdown_file(markdown_file):
    return Path('resources/md_files/'+markdown_file).read_text()

# read data for EDA
df_data = pd.read_csv('economic_data_for_EDA.csv')

# --------------------------- Main App Declaration -------------------------
# Main app/App declaration
def main():

    # Add background image
    add_bg_from_local('resources/background_image2.png') 
    
    # Add the Streamlit magic command to run the app    
    page_options = ['Home', 'Explore the Data', 'Data Analysis', 'About Team 9', 'FAQs', 'Feedback', 'Contact Us']
    app_page = st.sidebar.selectbox("CHOOSE AN OPTION", page_options)
   
    if app_page == 'Home':
        home_page()
    # pass


#----------------------------- Explore the Data ----------------------------------
    if app_page == "Explore the Data":
        st.title("Explore the Data")
        explore_data_page()



#----------------------------- Data Analysis ----------------------------------
    if app_page == "Data Analysis":
        st.title("Data Analysis and Visualization")

        st.subheader("GDP Trends Over the Years")
        # INTERACTIVE LINE PLOT - GDP Trends for Different Provinces Over the Years
        # Create the line plot
        fig = px.line(df_data, x='year', y='gdpr', color='province', title='GDP Trends by Province (2008-2019)')

        # Customize the hover information
        fig.update_traces(mode='lines+markers', hovertemplate='%{x}: %{y} GDP<br>')

        # Set the axis titles and legend title
        fig.update_layout(xaxis_title='Year', yaxis_title='GDPR', legend_title='Province')        
        fig.update_layout(plot_bgcolor='white')  # Change the background color- color options - 'lightgray'

        # Set grid lines to show
        fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        st.plotly_chart(fig) # Display the plot in the Streamlit app

        st.write("-------")

        st.subheader("province_wise Analysis")
        # INTERACTIVE DONUT CHART
        data = df_data.copy()        
        total_gdp = data['gdpr'].sum() # Calculate the total GDP for all provinces        
        data['percentage_gdp'] = (data['gdpr'] / total_gdp) * 100 # Calculate the percentage GDP for each province
        fig = go.Figure(data=[go.Pie(labels=data['province'], values=data['percentage_gdp'], hole=.3)]) # Create a pie chart
        fig.update_layout(title_text='Percentage GDP Distribution by Province')
        st.plotly_chart(fig)

        st.write("------")

        st.subheader("Sector_wise Analysis")


#------------------------- About Quantum Analytics -----------------------------
    if app_page == 'About Team 9':
        st.title("About Team 9")
        
        # About us expander
        expand = st.expander("Who Are We?")
        about_markdown = read_markdown_file("about_us.md")
        expand.markdown(about_markdown, unsafe_allow_html=True)

        # Photo of team
        st.subheader("Meet the Team")
        st.image("resources/team_photos/all_img.png")

        # Team contact details
        st.subheader("Team Contact details")
        st.write('    🖋️ Obinna Ekenonu - https://www.linkedin.com/in/obinna-ekenonu-5973a514/')
        st.write('    🖋️Chidinma Madukife - https://www.linkedin.com/in/chidinma-madukife/')
        st.write('    🖋️Kgotso Makhalimele - https://www.linkedin.com/in/kgotso-makhalimele/')
        st.write('    🖋️Tolulope Adeleke - https://www.linkedin.com/in/tolulope-adeleke-3330a6240/')
        st.write('    🖋️Samuel Olaniyi -' )


#------------------------- FAQs -----------------------------
    if app_page == "FAQs":
        st.title("FAQs")

        faq1 = st.expander("Q: What is the purpose of this app?")
        faq1.write("A: The app aims to predict and analyze Gross Domestic Product (GDP) trends, providing insights into economic performance and trends for different regions")

        faq2 = st.expander("Q: How accurate are the GDP predictions?")
        faq2.write("A:  The predictions are based on historical data and models. While efforts are made to provide accurate predictions, they should be considered as estimates.")

        faq3 = st.expander("Q: What shape file formats are supported")
        faq3.write("A: We currently support shape files in the SHP format. You can choose a shape file by clicking on the **Choose Shape File** section and selecting the desired file.")

        faq4 = st.expander("Q: What is the purpose of the GDP Prediction App?")
        faq4.write("A: Our GDP Prediction App utilizes satellite nighttime images and advanced machine learning algorithms to provide rapid and accurate GDP predictions. It helps businesses, policymakers, and researchers anticipate economic trends and make informed decisions.")

        faq5 = st.expander("Q: How often is the app updated with new data?")
        faq5.write("A: The GDP prediction is updated on a regular basis, typically on a monthly or quarterly basis, depending on the availability of new data. We strive to provide the most up-to-date and accurate predictions for our clients.")

        faq6 = st.expander("Q: Can I download or share the visualizations?")
        faq6.write("A: Currently, the app doesn't support downloading or sharing directly. However, you can take screenshots for sharing or download images using external tools.")

        faq6 = st.expander("Q: How secure is the data I upload to the app?")
        faq6.write("A: We take data security seriously. The data you upload to the app is stored securely and treated with strict confidentiality. We implement industry-standard security measures to protect your data and ensure it is accessed only by authorized personnel.")

        faq7 = st.expander("Q: Can I customize the visualization of the data analysis results?")
        faq7.write("A: Yes, our data analysis and visualization services offer customization options. You can specify your preferences for visualizations, including chart types, colors, labels, and other parameters, to ensure the results align with your specific requirements and branding.")

        faq8 = st.expander("Q: Can I explore satellite images in the app")
        faq8.write("A: Yes, you can explore satellite images and night light intensity data in the **Explore Data** section. Select the country, province, year, and click **Explore Data** to view the images.")

        faq9 = st.expander("Q: What data sources are used for GDP predictions?")
        faq9.write("A: The app uses historical GDP data for different regions. You can find more details on the specific data sources in the **About** section.")

        faq10 = st.expander("Q: Is there a dark mode available?")
        faq10.write("A: Yes, the app supports dark mode. You can switch between light and dark mode using the theme options.")


#--------------------------- Feedback -------------------------------
    if app_page == "Feedback":
        st.title("Feedback")
        # Feedback form
        survey = ss.StreamlitSurvey()
		# Likert scale
        slider = survey.select_slider(
			"How satisfied are you with this app?", options=["Very Dissatisfied", "Somewhat Dissatisfied", "Neutral", "Somewhat Satisfied", "Very Satisfied"], id="Q1"
		)
        feedback_text = survey.text_area("We would love to hear your feedback and suggestions",id="Q2")
        feedback_button = st.button("Submit Feedback")
        if feedback_button:
            if feedback_text and slider:
        # Save the feedback to a file, database, or process it as needed
                st.success("Thank you for your feedback!")
                
                feedback = survey.to_json()
                
                with open("resources/feedback.txt", "a") as f:
                    f.write(feedback)
                    f.write("\n")
            else:
                st.warning("Please enter your feedback before submitting.")


#------------------------- Contact Us -----------------------------
    if app_page == 'Contact Us':
        st.title("Contact Us")

        # Contact us section with map
        contact = st.expander("Contact Us")
        contact.write("Freedom way, Lekki Phase I 106104, Lekki, Lagos")
        contact.write("+234 083 000 0000")
        # Make a map with our fictional location
        df = pd.DataFrame(
        [[6.458985, 3.601521]],
        columns=['lat', 'lon'])

        contact.map(df,13)
        

if __name__ == '__main__':
    main()
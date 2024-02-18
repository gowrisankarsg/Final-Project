import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
import easyocr
from joblib import load
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
import spacy
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

data = pd.read_csv("classification_data.csv")


def classification():
    st.write("<h4 style='text-align:center;  font-weight:bolder;'>ECOMMERCE PREDICTION APPLICATION</h4>",unsafe_allow_html=True)
    tab1,tab2 = st.tabs(["PREDICTION","REPORTS"])
    channelgrouplist = list(data['channelGrouping'].unique())
    channelgrouplist.sort()
    devices = list(data['device_deviceCategory'].unique())
    devices.sort()
    regions = list(data['geoNetwork_region'].unique())
    regions.sort()
    sources = list(data['latest_source'].unique())
    sources.sort()
    e_medium = list(data['earliest_medium'].unique())
    e_medium.sort()
    l_medium = list(data['latest_medium'].unique())
    l_medium.sort()
    with tab1:
        with st.form("form1"):
            col1,col2 = st.columns(2)
            with col1:
                channel = st.selectbox("**Select the Channel**",channelgrouplist,index=None,placeholder="Please Selct")
                device = st.selectbox("**Select the Device**",devices,index=None,placeholder="Please Select")
                region = st.selectbox("**Select the Region**",regions,index=None,placeholder="Please Select")
                source = st.selectbox("**Select the Source**",sources,index=None,placeholder="Please Select")
                emedium = st.selectbox("**Select the Earliest Medium**",e_medium,index=None,placeholder="Please Select")
                lmedium = st.selectbox("**Select the Latest Medium**",l_medium,index=None,placeholder="Please Select")

            with col2:
                count_hit = st.text_input("**Enter Count hit range (0-20000)**")
                his_ses_page = st.text_input("**Enter Historic Session Page range (0-10000)**")
                average_visit = st.text_input("**Enter Average Vist Time range (0-100)**")
                time_on_site = st.text_input("**Enter Time On Site range (0-30000)**")
                trans_rev = st.text_input("**Enter Transaction Revenue range (0-1000000000)**")
                pred_but = st.form_submit_button("PREDICTION")
        
        if pred_but:
            if (channel != None) and (device != None) and (region != None) and (source != None) and (emedium != None) and (lmedium != None):
                counthit = float(count_hit)
                channels = int(channelgrouplist.index(channel))
                device_cat = int(devices.index(device))
                place = int(regions.index(region))
                his_page = float(his_ses_page)
                avg_time = float(average_visit)
                src = int(sources.index(source))
                em = int(e_medium.index(emedium))
                lm = int(l_medium.index(lmedium))
                tos = float(time_on_site)
                tr = float(trans_rev)
                features = {
                    "count_hit":float(counthit),
                    "channelGrouping":channels,
                    "device_deviceCategory":device_cat,
                    "geoNetwork_region":place,
                    "historic_session_page":float(his_page),
                    "avg_visit_time":float(avg_time),
                    "latest_source":src,
                    "earliest_medium":em,
                    "latest_medium":lm,
                    "time_on_site":float(tos),
                    "transactionRevenue":float(tr)
                }
                
                
                feature = pd.DataFrame([features])
                
                model = load("DTCModel.joblib")
                prediction = model.predict(feature)
                
                if prediction[0] == 0:
                    st.warning("NOT CONVERT")
                if prediction[0] == 1:
                    st.success("CONVERT")

    with tab2:
        st.write("<h4 style='text-align:center;  font-weight:bolder;'>MODEL REPORTS</h4>",unsafe_allow_html=True)
        reportdf = pd.read_csv("report.csv")
        reportdf.drop("Unnamed: 0",axis=1,inplace=True)
        st.dataframe(reportdf)

        st.subheader("Plot Representation of the Report Dataframe")
        fig = px.bar(reportdf,x='Model name',y=['Accuracy','Precision','Recall','F1_score'],barmode='group')
        st.plotly_chart(fig)

        st.subheader("LogisticRegression ROC Curve")
        st.image("LR_roc_curve.jpg")

        st.subheader("LDA ROC Curve")
        st.image("LDA_roc_curve.jpg")

        st.subheader("DecisionTreeClassifier ROC Curve")
        st.image("DT_roc_curve.jpg")

        st.subheader("RandomForestClassifier ROC Curve")
        st.image("RF_roc_curve.jpg")




def eda():
    st.write("<h4 style='text-align:center;  font-weight:bolder;'>EXPLORATORY DATA ANALYSIS</h4>",unsafe_allow_html=True)
    code = """
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.express as px"""
    st.subheader("Import Required Packages")
    st.code(code,language='python')

    # Read data
    st.subheader("Read Data")
    st.dataframe(data)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.table(data.describe())

    

    # Finding Empty Values
    st.subheader("Finding Null Values")
    nulldata = data.isnull().sum()
    nulldf = pd.DataFrame({"feature":nulldata.index, "null count": nulldata.values})
    st.dataframe(nulldf)
    
    # Find Duplicates
    #st.subheader("Find Duplicates")
    #duplicate = data.duplicated().sum()
    #st.write("Duplicate Counts : " , duplicate)
    
    # Find Sparcity
    st.subheader("Find Sparcity")
    col_zero = []
    for i in data.columns:
        perc_zero = ((data[i]==0).mean()*100).round(2)
        col_zero.append((i,perc_zero))
    zero_df = pd.DataFrame(col_zero,columns=["features",'zero_perc']).sort_values('zero_perc',ascending=False)
    st.dataframe(zero_df)
    
    # Univariate Analysis
    st.subheader("Univariate Analysis")

    # Histogram
    selected_column = st.selectbox("Select a column for histogram:", data.columns)
    fig_hist = px.histogram(data, x=selected_column, title=f'Histogram of {selected_column}')
    st.plotly_chart(fig_hist)

    # Boxplot
    fig_box = px.box(data, x=selected_column, title=f'Boxplot of {selected_column}')
    st.plotly_chart(fig_box)

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")

    # Scatter plot
    x_axis = st.selectbox("Select X-axis for scatter plot:", data.columns)
    y_axis = st.selectbox("Select Y-axis for scatter plot:", data.columns)
    fig_scatter = px.scatter(data, x=x_axis, y=y_axis, title=f'Scatter Plot: {x_axis} vs {y_axis}')
    st.plotly_chart(fig_scatter)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    st.image('correlation.jpg')

    # outlier detection
    st.subheader("Outlier Detection")
    box = px.box(data[['count_hit','historic_session_page','avg_visit_time','time_on_site','transactionRevenue']]) 
    st.plotly_chart(box)

    def remove_outlier(col):
        sorted(col)
        Q1,Q3=np.percentile(col,[25,75])
        IQR=Q3-Q1
        lower_range= Q1-(1.5 * IQR)
        upper_range= Q3+(1.5 * IQR)
        return lower_range, upper_range
    x = data[['count_hit','historic_session_page','avg_visit_time','time_on_site','transactionRevenue']]
    for column in x.columns:
        if x[column].dtype != 'object':
            lr,ur=remove_outlier(x[column])
            x[column]=np.where(x[column]>ur,ur,x[column])
            x[column]=np.where(x[column]<lr,lr,x[column])

    st.subheader("Removed Outtlier")
    box1 = px.box(x) 
    st.plotly_chart(box1)
    

    st.subheader("Location based Plot")
    opt = st.selectbox("**Select Target Value**",[0,1],index=None,placeholder="Please Select")
    if opt == None:
        st.map(data,latitude="geoNetwork_latitude",longitude="geoNetwork_longitude")
    if opt != None:
        Location_plot = data[data['has_converted'] == int(opt)]
        st.map(Location_plot,latitude="geoNetwork_latitude",longitude="geoNetwork_longitude")


def image():
    st.write("<h4 style='text-align:center;  font-weight:bolder;'>IMAGE PROCESSING</h4>",unsafe_allow_html=True)
    upload_file = st.file_uploader('Choose a Image File', type=['png','jpg','webp'])

    if upload_file is not None:
        upload_image = np.asarray(Image.open(upload_file))
        u1 = Image.open(upload_file)
        
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Read Original Image")
            st.image(upload_image,)
            width =st.number_input("**Enter Width**",value=(u1.size)[0])
        
        with col2:
            graysclae = u1.convert("L")                   
            st.subheader("Gray Scale Image")
            st.image(graysclae)
            height =st.number_input("**Enter Height**",value=(u1.size)[1])

        with col1:
            resize_image = u1.resize((int(width),int(height)))
            st.subheader("Resize Image")
            st.image(resize_image)
            radius =st.number_input("**Enter radius**",value=1)
            blur_org = u1.filter(ImageFilter.GaussianBlur(radius = int(radius) ))
            st.subheader("Blurring with Original Image")
            st.image(blur_org)
            blur_gray = graysclae.filter(ImageFilter.GaussianBlur(radius = int(radius) ))
            st.subheader("Blurring with Gray Scale Image")
            st.image(blur_gray)
            threshold = st.number_input("**Enter Threshold**",value=100)
            threshold_image = u1.point(lambda x: 0 if x< threshold else 255)
            st.subheader("Threshold Image")
            st.image(threshold_image)
            flip = st.selectbox("**Select Flip**",["left-right",'top-bottom'])
            st.subheader("Flipped Image")
            if flip == "left-right":
                st.image(u1.transpose(Image.FLIP_LEFT_RIGHT))
            if flip == 'top-bottom':
                st.image(u1.transpose(Image.FLIP_TOP_BOTTOM))
            brightness =st.number_input("**Enter Brightness**",value=1)
            st.subheader("Brightness Image")
            st.image((ImageEnhance.Brightness(u1)).enhance(int(brightness)))


        with col2:
            mirror_image = ImageOps.mirror(u1)
            st.subheader("Mirror Image")
            st.image(mirror_image)
            contrast =st.number_input("**Enter contrast**",value=1)
            contrast_org = ImageEnhance.Contrast(blur_org)
            st.subheader("Contrast with Original Image")
            st.image(contrast_org.enhance(int(contrast)))
            contrast_gray = ImageEnhance.Contrast(blur_gray)
            st.subheader("Contrast with Gray Scale Image")
            st.image(contrast_gray.enhance(int(contrast)))
            rotation =st.number_input("**Enter Rotation**",value=180)
            st.subheader("Rotation Image")
            st.image(u1.rotate(int(rotation)))
            sharpness =st.number_input("**Enter Sharness**",value=1)
            st.subheader("Sharpness Image")
            st.image((ImageEnhance.Sharpness(u1)).enhance(int(sharpness)))
            image_type = st.selectbox("**Select Image**",["Original image",'Gray Scale Image',"Blur Image","Threshold Image","Sharpness Image","Brightness Image"])
            
            if image_type == "Original image":
                st.subheader("Edge Detection with Original Image")
                st.image(u1.filter(ImageFilter.FIND_EDGES))
            if image_type == 'Gray Scale Image':
                st.subheader("Edge Detection with Grayscale Image")
                st.image(graysclae.filter(ImageFilter.FIND_EDGES))
            if image_type == "Blur Image":
                st.subheader("Edge Detection with Blur Original Image")
                st.image(blur_org.filter(ImageFilter.FIND_EDGES))
            
            if image_type == "Threshold Image":
                st.subheader("Edge Detection with Threshold Image")
                st.image(threshold_image.filter(ImageFilter.FIND_EDGES))
            if image_type == "Sharpness Image":
                st.subheader("Edge Detection with Sharpness Image")
                st.image(((ImageEnhance.Sharpness(u1)).enhance(int(sharpness))).filter(ImageFilter.FIND_EDGES))
            if image_type == "Brightness Image":
                st.subheader("Edge Detection with Brightness Image")
                st.image(((ImageEnhance.Brightness(u1)).enhance(int(brightness))).filter(ImageFilter.FIND_EDGES))
        
        reader = easyocr.Reader(['en'])
        bounds = reader.readtext(upload_image)
        if bounds != '':
            st.subheader("Extracted Text")
            file_name = upload_file.name
            if file_name == '1.png':
                
                address,city = map(str,(bounds[6][1]).split(', '))
                state,pincode = map(str,(bounds[8][1]).split())
                image1_data = {
                    'Company': bounds[7][1]+' '+bounds[9][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[2][1],
                    'Email': bounds[5][1],
                    'URL': bounds[4][1],
                    'Area':address[0:-1],
                    'City': city[0:-1],
                    'State':state,
                    'Pincode': pincode
                }
                st.json(image1_data)
                    
            if file_name == '2.png':
                state,pincode = map(str,(bounds[9][1]).split())
                image2_data = {
                    'Company': bounds[8][1]+' '+bounds[10][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[2][1],
                    'Email': bounds[3][1],
                    'URL': bounds[4][1]+'.'+bounds[5][1],
                    'Area': (bounds[6][1]+' '+bounds[11][1])[0:-2],
                    'City': (bounds[7][1])[0:-1],
                    'State':state,
                    'Pincode': pincode
                }
                st.json(image2_data)
                

            if file_name == '3.png':
                address,city = map(str,(bounds[2][1]).split(', '))
                state,pincode = map(str,(bounds[3][1]).split())
                image3_data = {
                    'Company': bounds[7][1]+' '+bounds[8][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[4][1],
                    'Email': bounds[5][1],
                    'URL': bounds[6][1],
                    'Area': address[0:-1],
                    'City': city[0:-1],
                    'State':state,
                    'Pincode': pincode
                }
                st.json(image3_data)


            if file_name == '4.png':
                area,city,state = map(str,(bounds[2][1]).split(', '))
                image4_data = {
                    'Company': bounds[6][1]+' '+bounds[8][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[4][1],
                    'Email': bounds[5][1],
                    'URL': bounds[7][1],
                    'Area': area[0:-1],
                    'City': city,
                    'State':state,
                    'Pincode': bounds[3][1]
                }
                st.json(image4_data)
                

            if file_name == '5.png':
                area,city,state = map(str,(bounds[2][1]).split(', '))
                image5_data = {
                    'Company': bounds[7][1],
                    'Card_holder_name': bounds[0][1],
                    'Desination': bounds[1][1],
                    'Mobile': bounds[4][1],
                    'Email': bounds[5][1],
                    'URL': bounds[6][1],
                    'Area': area[0:-1],
                    'City': city,
                    'State':state,
                    'Pincode': bounds[3][1]
                }
                st.json(image5_data)
        

def nlp():
    st.write("<h4 style='text-align:center;  font-weight:bolder;'>NATURAL LANGUAGE PROCESSING</h4>",unsafe_allow_html=True)
    
    col1,col2 = st.columns(2)
    with col1:
        text = st.text_input("**Enter Bunch of Text**")
    
    with col2:
        process = st.button("START PROCESSING")


    if process:
        # original text
        st.subheader("Original Text :")
        st.text_area("Original Text",text)

        # Text Minning / Cleaning
        st.write("<h5 style='text-align:center;  font-weight:bolder;'>TEXT MINNING / CLEANING</h5>",unsafe_allow_html=True)

        # convert lowercase
        st.subheader("Convet Lowercase :")
        text_low = (str(text)).lower()
        st.text_area("**Text of Lowercase**",text_low)

        # remove url
        def remove_urls(text):
            
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            
            cleaned_text = re.sub(url_pattern, '', text)
            return cleaned_text
        text_url = remove_urls(text_low)
        st.subheader("Remove URLs :")
        st.text_area("**Text of Removed URLs**",text_url)

        # remove punctuation
        def remove_punctuation(text):
            
            punctuation_pattern = re.compile(r'[^\w\s]')
            
            cleaned_text = re.sub(punctuation_pattern, '', text)
            return cleaned_text
        text_punc = remove_punctuation(text_url)
        st.subheader("Remove Punctuation :")
        st.text_area("**Text of Removed Punctuation**",text_punc)

        # remove stopwords
        def remove_stopwords(text):
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(text)
            filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
            return ' '.join(filtered_text)
        text_stop = remove_stopwords(text_punc)
        st.subheader("Remove Stopwords :")
        st.text_area("**Text of Removed Stopwords**",text_punc)

        # remove numerals
        def remove_numerals(text):
            
            pattern = r'\d+'
            
            text_without_numerals = re.sub(pattern, '', text)
            return text_without_numerals
        text_not_num = remove_numerals(text_stop)
        st.subheader("Remove Numerals :")
        st.text_area("**Text of Removed Numerals**",text_not_num)

        # tokenization
        text_token = word_tokenize(text_not_num)
        st.subheader("Tokenization :")
        tokendf = pd.DataFrame({"Tokenized_word":text_token})
        st.dataframe(tokendf)

        # stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in text_token]
        stem = {
            "Tokenized_words": text_token,
            "Stemmed_words": stemmed_tokens
        }
        stemdf = pd.DataFrame(stem)
        st.subheader("Stemming :")
        st.dataframe(stemdf)

        # lemmatizing
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in text_token]
        lem = {
            "Tokenized_words": text_token,
            "Stemmed_words": lemmatized_tokens
        }
        lemdf = pd.DataFrame(stem)
        st.subheader("Lemmatizing :")
        st.dataframe(lemdf)

        # Named Entity Recognition (NER)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        ners = [entity.text for entity in doc.ents]
        nertext = []
        nerlabel = []
        for ent in doc.ents:
            nertext.append(ent.text)
            nerlabel.append(ent.label_)
        ner = {
            "NER_text": nertext,
            "NER_label": nerlabel
        }
        nerdf = pd.DataFrame(ner)
        st.subheader("Named Entity Recognition (NER)")
        st.dataframe(nerdf)

        # keyword extraction
        
        all_keywords = text_token + ners 
        keyword_freq = Counter(all_keywords)
        top_keywords = keyword_freq.most_common(5)
        word = []
        freq = []
        for key in top_keywords:
            word.append(key[0])
            freq.append(key[1])
        key = {
            "word": word,
            'freq': freq
        }
        keydf = pd.DataFrame(key)
        
        st.subheader("Keyword Extraction")
        st.dataframe(keydf)

        # sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        positive_words = []
        negative_words = []
        neutral_words = []

        for item in text_token:
            score = analyzer.polarity_scores(item)['compound']
            if score > 0:
                positive_words.append(item)
            elif score < 0:
                negative_words.append(item)
            else:
                neutral_words.append(item)
        
        st.write("<h4 style='text-align:center;  font-weight:bolder;'>SENTIMENT ANALYSIS</h4>",unsafe_allow_html=True)
        col5,col6,col7 = st.columns(3)
        with col5:
            st.subheader("Positive")
            pdf = pd.DataFrame({'Positive_words':positive_words})
            st.dataframe(pdf)
        
        with col6:
            st.subheader("Negative")
            ndf = pd.DataFrame({'Negative_words':negative_words})
            st.dataframe(ndf)
        
        with col7:
            st.subheader("Neutral")
            ntdf = pd.DataFrame({'Neutral_words':neutral_words})
            st.dataframe(ntdf)

        # wordcloud
            
        st.write("<h4 style='text-align:center;  font-weight:bolder;'>WORDCLOUD</h4>",unsafe_allow_html=True)
        
        positive_word = " ".join(positive_words)
        pw = WordCloud().generate(positive_word)
        pwf = px.imshow(pw,title="POSITIVE")
        st.plotly_chart(pwf)

        
        negative_word = " ".join(negative_words)
        nw = WordCloud().generate(negative_word)
        nwf = px.imshow(nw,title="NEGATIVE")
        st.plotly_chart(nwf)

        
        neutral_word = " ".join(neutral_words)
        ne = WordCloud().generate(neutral_word)
        nef = px.imshow(ne,title="NEUTRAL")
        st.plotly_chart(nef)

def recommendation():
    st.write("<h4 style='text-align:center;  font-weight:bolder;'>PRODUCT RECOMMENDATION SYSTEM</h4>",unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        product = st.text_input("**Enter Product Name**")
    
    with col2:
        rec_num = st.number_input("**Enter a Number**",value=5)

    reocommend = st.button("**GET RECOMMEND**")
    st.write("<h4 style='text-align:center;  font-weight:bolder;'>RECOMMENDED PRODUCTS</h4>",unsafe_allow_html=True)

    laptop = pd.read_csv("laptop.csv",encoding="unicode_escape").apply(lambda x: x.astype(str).str.lower())
    products = list(laptop.name.unique())

    if reocommend:
        def recommend_similar_products(product_query, product_names, top_n=int(rec_num)):
            # Add the queried product to the list
            product_names.append(product_query)

            # Vectorize product names
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(product_names)

            # Compute cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Get indices of similar products
            query_index = len(product_names) - 1
            similar_indices = cosine_sim[query_index].argsort()[:-top_n-2:-1]

            # Display top n similar products
            similar_products = [(cosine_sim[query_index][i], product_names[i]) for i in similar_indices]
            return similar_products[1:]

        similar_products = recommend_similar_products(product, products)
        for similarity, Product in similar_products:
            st.write(f"<ul><li>{Product}</li></ul>",unsafe_allow_html=True) 



st.set_page_config(layout='wide')

st.write("<h2 style='text-align:center; margin-top:-60px; font-weight:bolder; '>FINAL PROJECT</h2>",unsafe_allow_html=True)

option = option_menu(
    menu_title=None,
    options=['CLASSIFICATION','EDA','IMAGE','NLP','RECOMMENDATION'],
    orientation='horizontal',
    styles={
                "container": {"padding": "0!important", "background-color": "#fafafa", "width": '100%', },
                "icon": {"color": "orange", "font-size": "11px"},
                "nav-link": {
                    "font-size": "11px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#525ceb",
                },
                "nav-link-selected": {"background-color": "green"},
            },
    )

st.markdown(
        """
        <style>
            .st-ax{
                background-color: lightblue;
                
            }

            .stTextInput input{
                background-color: lightblue;
            }
            
            .stNumberInput input{
                background-color: lightblue;
            }

            .stButton>button{
                background-color: #01A982;
                color:#ffffff;
                width:5em;
                color:#ffffff;
                width: 255px;
                transition-duration: 0.4s;
                margin: 10px 70px;
                border-radius: 25px;
                border-radius: 5px 25px;
                padding: 1px;
                box-shadow: 2px 2px 5px gray;
                transition: color 0.3s ease-in-out;
                animation: spin 2s linear infinite;
                
            }

            

        </style>
        """
    ,unsafe_allow_html=True
    )

if option == "CLASSIFICATION":
    classification()

if option == "EDA":
    eda()

if option == "IMAGE":
    image()

if option == "NLP":
    nlp()

if option == "RECOMMENDATION":
    recommendation() 
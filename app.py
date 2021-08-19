import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



st.title("Simple Plagiarism Detector")


files = []
uploaded_files = st.file_uploader("Choose files" , accept_multiple_files = True)
for file in uploaded_files:
    files.append(file)

if len(files) < 2:
    st.markdown("Please select atleast two files")
else:
    documents = [file.read() for file in files]


    def vectorize(text):
        tfidf = TfidfVectorizer()
        X = tfidf.fit(text)
        X = tfidf.transform(text)
        vectors = X.toarray()
        return vectors


    def similarity(doc1 , doc2):
        return cosine_similarity([doc1 , doc2])


    vectors = vectorize(documents)
    file_vectors = list(zip(files, vectors))

    final_result = []
    def check_plagiarism():
        global file_vectors
        temp_vect = file_vectors.copy()
        for file1 , text_vector_1 in file_vectors:
            now_detecting = temp_vect.index((file1 , text_vector_1))
            del temp_vect[now_detecting]
            for file2 , text_vector_2 in temp_vect:
                similarity_score = similarity(text_vector_1 , text_vector_2)[0][1]
                file_pair = (file1 , file2)
                plagiarism_score = (file_pair[0] , file_pair[1] , similarity_score)
                final_result.append(plagiarism_score)
        return final_result


    if st.button("Check"):
        for result in check_plagiarism():
            st.write("Similarity Score of ",result[0].name , " and " , result[1].name , " is : " , result[2])


st.markdown("_____________________________________________________________________________________________________")
st.write("  \n  ")
st.markdown("Copyright © 2021 | Vampire_Questt | Deepak Rajveer Singh | All rights reserved")


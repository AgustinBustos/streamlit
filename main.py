import streamlit as st
import pandas as pd
from datetime import datetime
import SMjournal as smj

#change name of stats

# smjournal dont show option
# change beta
# correlation thresh

st.set_page_config(page_title="Stats",layout="wide",page_icon=None)

def select_cols(cols):
  dims=["Geographies","Weeks"]
  weight=[i for i in cols if ".LEQHHwgt" in i][0]
  y_col=[i for i in cols if ".LEQHHDEP" in i][0]
  x_cols=[i for i in cols if i not in dims+[weight,y_col]]
  return dims, x_cols, y_col, weight



st.title('Stats')
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Collinearity Tests", "Feature Selection", 'Meta Regression'])

with tab1:
   
    holder = st.empty()
    uploaded_file = holder.file_uploader("Choose a CSV to upload, make sure to have Weeks, Geographies, the dependent variable and weight")
    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        df['Weeks']=pd.to_datetime(df['Weeks'])
        df=df.sort_values('Weeks')
     

        st.text("")
        st.text("")
        geos = st.multiselect(
                'Select the Geographies to use',
                list(df['Geographies'].unique()),
                list(df['Geographies'].unique()))
        first_day=df['Weeks'].unique()[0]
        last_day=df['Weeks'].unique()[-1]
   
       
        weeks = st.slider(
            "Select dates.",
            value=(first_day.to_pydatetime(),last_day.to_pydatetime())
        )
        

        df=df.loc[df['Geographies'].isin(geos)]
        df=df.loc[(df['Weeks']>weeks[0]) & (df['Weeks']<weeks[1])]
        st.write(df)
        holder.empty()

        
with tab2:

    if uploaded_file is not None:
        dims, x_cols, y_col, weight=select_cols(df.columns)
        selected = st.multiselect(
                'Select the Columns to use',
                x_cols,
               x_cols)
        filt = st.number_input('Insert a Correlation filter')
        if st.button('Run'):

            clus, fig, plt, corrplot=smj.collinearity_test(df[['Weeks']+selected],streamlit=st,filter=filt)
            st.pyplot(corrplot)
            st.plotly_chart(fig, use_container_width=True)
            st.pyplot(plt)

    
with tab3:
    st.title("Collinearity Tests")
with tab4:
    st.title("Collinearity Tests")

   



    






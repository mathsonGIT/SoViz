import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import plotly.express as px



def process_model(initial_vector, cost_matrix, step):
    rez = []
    size = initial_vector.shape[0]
    initial_matrix = np.zeros(initial_vector.shape)
    for i in range(step):
        rez.append(initial_matrix)
        initial_matrix = initial_matrix @ cost_matrix + initial_vector
    return (rez)

def make_graph_from_dataframe(df):
    graph = graphviz.Digraph()
    columns = df.columns
    for i in columns:
        for j in columns:
            if np.round(df.loc[i,j] , 3) != 0.000:
                graph.edge(str(i),str(j), str(df.loc[i,j]))
    return(graph)

def hide_streamlit_info():
   


    hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
    
    
    
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
            
def change_file_uploader():
    file_drop_set = {
    
        "button": "Загрузите файл в формате csv",
        "instructions": "Перетащите файл",
        "limits": "Размср файла ограничен 200 МБ",
    }
    hide_label = (
    """
    <style>
    div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"] {
       color:white;
    }
    div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"]::after {
        content: "BUTTON_TEXT";
        color:black;
        display: block;
        position: absolute;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>span {
       visibility:hidden;
    }
    
    div[data-testid="stFileDropzoneInstructions"]>div>span::after {
       content:"INSTRUCTIONS_TEXT";
       visibility:visible;
       display:block;
    }
     div[data-testid="stFileDropzoneInstructions"]>div>small {
       visibility:hidden;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>small::before {
       content:"FILE_LIMITS";
       visibility:visible;
       display:block;
    }
    </style>
    """.replace("BUTTON_TEXT", file_drop_set.get("button")).replace("INSTRUCTIONS_TEXT", file_drop_set.get("instructions")).replace("FILE_LIMITS", file_drop_set.get("limits"))
    )
    st.markdown(hide_label, unsafe_allow_html=True)

def hide_sliders_info():
    hide_elements = """
        <style>
            div[data-testid="stSliderTickBarMin"],
            div[data-testid="stSliderTickBarMax"] {
                display: none;
            }
        </style>
    """
    st.markdown(hide_elements, unsafe_allow_html=True)


def initial_vibb_slider(size: int, names: list):
    data = np.zeros(size)
    st.markdown('Установите начальные возмущения')
    with st.container():
        slider_cols = st.columns(size)
        for i in range(size):
            with slider_cols[i]:
                st.write(names[i])
                data[i] = st.slider(label=f'factor{i}', min_value=-1.0, max_value=1.0, value=0.0, step = 0.1, label_visibility='hidden')
    hide_sliders_info()
    return(data)




def table_of_sliders(size : int, names : list):
    data_array = np.zeros((size, size))
    st.markdown('Установите связи факторов')
    with st.container():
        slider_cols = st.columns(size)
        
        for i in range(size):
            with slider_cols[i]:
                #st.write(names[i])
                
                for j in range(size):
                    data_array[j, i] = st.slider(label = f'фактор{i} на фактор {j}', min_value=-1.0, max_value=1.0, value=0.1, step=0.1, label_visibility = 'hidden', key = f'factor{i}_{j}')
    hide_sliders_info()

    return data_array

def old_model():
      """ st.markdown('Установите начальные возмущения')
    if 'initial_vibb' not in st.session_state:
        initial_vibb = pd.DataFrame(data = np.zeros((1, num_factors)), columns = factors_name)
        initial_vibb = st.data_editor(initial_vibb, key="initial_edit", use_container_width=True)

    st.markdown('Установите первоначальные связи процессов ')
    if 'initial_matrix1' not in st.session_state:
        df = pd.DataFrame(index=factors_name,columns=factors_name, data=np.zeros((num_factors, num_factors)))
        initial_matrix = st.data_editor(df, key="model_edit", use_container_width=True)
    else:
        initial_matrix = st.data_editor(initial_matrix1, key="model_edit", use_container_width=True)


    if st.button('Проанализировать'):
        new_df = process_model(initial_vector=initial_vibb.to_numpy(), cost_matrix=initial_matrix.to_numpy(), step = step_number)

    #st.write(new_df)
        st.markdown('Визуализация предсказаний моделируемого процесса')
        #st.line_chart(pd.DataFrame(data = new_df, columns=factors_name), x_label=f'Время({step_model})')
        fig = px.line(pd.DataFrame(data = new_df, columns=factors_name)).update_layout(xaxis_title=f"Время({step_model})", yaxis_title="Прогноз")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('Когнитивная карта моделируемого процесса')
        graph = make_graph_from_dataframe(initial_matrix)

        st.graphviz_chart(graph) """

def holin_model():
    

    factors_name = []
    hide_streamlit_info()
    with st.sidebar:
        st.title('SoViz')
        st.caption('Инструмент для создания и визуализации социально-экономических моделей')
        st.image('data/logo_holin1.png')
        #change_file_uploader()
        #uploaded_file = st.file_uploader(label = "Загрузите модель из файла", type = ['csv'])
        
        #if uploaded_file is not None:
        #    initial_matrix1 = pd.read_csv(uploaded_file, index_col=[0])
        #    st.session_state['initial_matrix1'] = initial_matrix1
        #    factors_name_past = initial_matrix1.columns.to_list()
        #else:
        #    if st.session_state.get('initial_matrix1') is not None:
        #        del st.session_state['initial_matrix1']
            
            

        
        st.markdown('Базовые наcтройки модели')
        if 'initial_matrix1' not in st.session_state:
            num_factors = st.slider("Выберите количество факторов в модели", 2, 10, 3)
            for i in range(num_factors):
                factors_name.append(st.text_input(label = f'Фактор №{i+1}', key = f'factor_{i+1}', value = f'Фактор №{i+1}'))
        else:
            pass
            #num_factors = len(factors_name_past)
            #for i in range(num_factors):
            #    factors_name.append(st.text_input(label = f'Фактор №{i+1}', key = f'factor_{i+1}', value = factors_name_past[i]))
    
        
        step_model = st.selectbox(label = 'Выберите шаг прогноза модели', options = ['Месяц', 'Квартал', 'Год'])
        step_number = st.number_input(label = 'Установите количество шагов', min_value=5, step=1)
    
    
    st.header('Моделирование и анализ')

  
    # Create a graph
    initial_vibb2 = initial_vibb_slider(num_factors, factors_name)
    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    initial_matrix2 = table_of_sliders(num_factors, factors_name)
    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.markdown('Визуализация предсказаний моделируемого процесса')
    #st.write(initial_vibb2)
    #st.write(initial_matrix2)
    new_data = process_model(initial_vector=initial_vibb2, cost_matrix=initial_matrix2, step = step_number)
    #st.write(new_data)
        #st.line_chart(pd.DataFrame(data = new_df, columns=factors_name), x_label=f'Время({step_model})')
    #new_data = process_model(initial_vector=initial_vibb2, cost_matrix=initial_matrix2, step = step_number)
    fig = px.line(pd.DataFrame(data = new_data, columns=factors_name)).update_layout(xaxis_title=f"Время({step_model})", yaxis_title="Прогноз")
    st.plotly_chart(fig, use_container_width=True)
    if st.button(label = 'Когнитивная карта моделируемого процесса'):
        st.markdown('Когнитивная карта моделируемого процесса')
        #df = pd.DataFrame(data = initial_matrix2, columns=factors_name)
        #st.write(df)
        graph = make_graph_from_dataframe(pd.DataFrame(data = initial_matrix2, columns=factors_name, index=factors_name))
        st.write(graph)
        #st.graphviz_chart(graph) 

if __name__ == "__main__":
   holin_model()

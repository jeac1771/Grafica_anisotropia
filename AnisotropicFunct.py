# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:12:08 2020

@author: Jorge Alvarez Correa
"""
import sys
import PySimpleGUI as sg
import numpy as np
import pandas as pd
import math

# ------------------------------- This is to include a matplotlib figure in a Tkinter canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm
import matplotlib.patches as mpatch

print('hola soy pablo y estoy hablando por git')

print('modificado desde la pagina GIT')

print('test32')

def table_example():
    sg.SetOptions(auto_size_buttons=True)
    filename = sg.PopupGetFile('filename to open', no_window=True, file_types=(("CSV Files", "*.csv"),))
    # --- populate table with file contents --- #
    if filename == '':
        sys.exit(69)
    data = []
    header_list = []
    button = sg.PopupYesNo('Does this file have column names already?')
    
    print(filename)
    
    if filename is not None:
        try:
            df= pd.read_csv(filename, sep=',', engine='python', header=None)  # Header=None means you directly pass the columns names to the dataframe
            clasificar_j= pd.read_csv(filename, sep=',') # para que sea leido por el algoritmo
            data = df.values.tolist()               # read everything else into a list of rows
            if button == 'Yes':                     # Press if you named your columns in the csv
                header_list = df.iloc[0].tolist()   # Uses the first row (which should be column names) as columns names
                data = df[1:].values.tolist()       # Drops the first row in the table (otherwise the header names and the first row will be the same)
            elif button == 'No':                    # Press if you didn't name the columns in the csv
                sg.PopupError('Need column names')
        except:
            sg.PopupError('Error reading file')
            sys.exit(69)
 
##################################################################################################################
##################################################################################################################
######################################### PLOT ANISOTROPIA OPEN PIT ##############################################

    # agregar a cada tipo un color diferente
    categories = np.unique(clasificar_j['TIPO'])
    colors = cm.rainbow(np.linspace(0, 1, len(categories)))
    colordict = dict(zip(categories, colors))  
    clasificar_j["Color"] = clasificar_j['TIPO'].apply(lambda x: colordict[x])


    def subparalelismo():
        
        favor = []
        favor.clear()
        encontra = []
        encontra.clear()
        no = []
        no.clear()
        
        global modified2, newj
    
        for i in range(len(clasificar_j['DIP'])):
        
            if clasificar_j['DIPDIR'][i] <= DIPDIRTALUD + SUBGRADOS and clasificar_j['DIPDIR'][i] >= DIPDIRTALUD - SUBGRADOS: #append a los que son subparalelos al talud
                favor.append([clasificar_j['SISTEMA'][i], clasificar_j['TIPO'][i], clasificar_j['Color'][i],
                              clasificar_j['DIP'][i], math.radians(clasificar_j['DIP'][i])])

        ###################    
            elif DIPDIRTALUD - SUBGRADOS < 0: #como tomar el rango de subparalelismo en terminos de azimmuth si es menor a 0

                if  360 + (DIPDIRTALUD - SUBGRADOS) <= clasificar_j['DIPDIR'][i] <= 360:
                    favor.append([clasificar_j['SISTEMA'][i], clasificar_j['TIPO'][i], clasificar_j['Color'][i],
                                  clasificar_j['DIP'][i], math.radians(clasificar_j['DIP'][i])])

                elif clasificar_j['DIPDIR'][i] <= DIPDIRTALUD+180+SUBGRADOS and clasificar_j['DIPDIR'][i] >= DIPDIRTALUD+180-SUBGRADOS:
                    encontra.append([clasificar_j['SISTEMA'][i], clasificar_j['TIPO'][i],clasificar_j['Color'][i],
                                     clasificar_j['DIP'][i], math.radians(clasificar_j['DIP'][i])])
                
            elif (DIPDIRTALUD + 180)> 360 or (DIPDIRTALUD + 180 + SUBGRADOS)> 360 or (DIPDIRTALUD + SUBGRADOS) > 360: #como tomar el rango de subparalelismo si el rango da mayor a 360

                if  0 <=  clasificar_j['DIPDIR'][i] <= (DIPDIRTALUD + SUBGRADOS) - 360:
                    favor.append([clasificar_j['SISTEMA'][i], clasificar_j['TIPO'][i], clasificar_j['Color'][i],
                                  clasificar_j['DIP'][i], math.radians(clasificar_j['DIP'][i])])

                elif clasificar_j['DIPDIR'][i] <= DIPDIRTALUD-180+SUBGRADOS and clasificar_j['DIPDIR'][i] >= DIPDIRTALUD-180-SUBGRADOS:
                    encontra.append([clasificar_j['SISTEMA'][i], clasificar_j['TIPO'][i],clasificar_j['Color'][i],
                                     clasificar_j['DIP'][i], math.radians(clasificar_j['DIP'][i])])
                    
                elif clasificar_j['DIPDIR'][i] <= 360 and clasificar_j['DIPDIR'][i] >= DIPDIRTALUD+180-SUBGRADOS:
                    encontra.append([clasificar_j['SISTEMA'][i], clasificar_j['TIPO'][i],clasificar_j['Color'][i],
                                     clasificar_j['DIP'][i], math.radians(clasificar_j['DIP'][i])])

        ####################

            elif clasificar_j['DIPDIR'][i] <= DIPDIRTALUD+180+SUBGRADOS and clasificar_j['DIPDIR'][i] >= DIPDIRTALUD+180-SUBGRADOS: #como tomar el rango si esta en sentido opuesto
                encontra.append([clasificar_j['SISTEMA'][i], clasificar_j['TIPO'][i],clasificar_j['Color'][i],
                                 clasificar_j['DIP'][i], math.radians(clasificar_j['DIP'][i])])

            else:
                no.append([clasificar_j['SISTEMA'][i], clasificar_j['DIP'][i]])

        # guardamos los append en un df   
        favordf = pd.DataFrame(favor, columns =['SISTEMAF','TIPOFAV', 'ColorF','DIPFAVOR', 'DIPRADSFAV'])
        contradf = pd.DataFrame(encontra, columns =['SISTEMAC','TIPOCONTRA','ColorC','DIPCONTRA', 'DIPRADSCON'])
        modified2 = pd.concat([favordf, contradf], axis=1)
        newj = modified2.drop(columns=['ColorF', 'DIPRADSFAV', 'ColorC', 'DIPRADSCON'])
    

    def draw_figure_w_toolbar(canvas, fig, canvas_toolbar): #esto es para poder agregar las figuras a la app
        if canvas.children:
            for child in canvas.winfo_children():
                child.destroy()
        if canvas_toolbar.children:
            for child in canvas_toolbar.winfo_children():
                child.destroy()
        figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
        figure_canvas_agg.draw()
        toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
        toolbar.update()
        figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


    class Toolbar(NavigationToolbar2Tk): #esto es para poder agregar las figuras a la app
        def __init__(self, *args, **kwargs):
            super(Toolbar, self).__init__(*args, **kwargs)


    def draw_figure(canvas, figure, loc=(0, 0)): #esto es para poder agregar las figuras a la app
        if canvas.children:
            for child in canvas.winfo_children():
                child.destroy()
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg
    
    ANCHOANISO = 10
    ANCHORADIANS = math.radians(ANCHOANISO)

    def plotanisotropia():
        
        plt.close(1)
        plt.figure(1)
        fig = plt.gcf()
        DPI = fig.get_dpi()
        ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True, facecolor='#d5de9c')
        fig.set_size_inches(304 * 2 / float(DPI), 404 / float(DPI))
    
        #### GRAFICAR BARRAS

        plt.bar(modified2['DIPRADSCON'].dropna() + math.radians(90), 1, width=ANCHORADIANS*2, bottom=0.0,
                color=modified2['ColorC'].dropna(), alpha = 0.7)
        plt.bar(math.radians(90) - modified2['DIPRADSFAV'].dropna(), 1, width=ANCHORADIANS*2, bottom=0.0,
                color=modified2['ColorF'].dropna(), alpha = 0.7)

        #### GRAFICAR LINEAS

        for i in range(len(modified2['DIPRADSCON'].dropna())):
            theta0 = np.deg2rad([modified2['DIPCONTRA'][i], modified2['DIPCONTRA'][i]]) + math.radians(90)
            R0 = [0,1]
            ax.plot(theta0, R0, lw=2, color = 'black')
        
    
        for i in range(len(modified2['DIPFAVOR'].dropna())):
            theta0 = math.radians(90) - np.deg2rad([modified2['DIPFAVOR'][i], modified2['DIPFAVOR'][i]])
            R0 = [0,1]
            ax.plot(theta0, R0, lw=2, color = 'black')
        
        #### GRAFICAR ANOTACIONES DE ÁNGULOS
    
        bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2)
        
        for i in range(len(modified2['DIPRADSCON'].dropna())):    
            ax.annotate(str(modified2['DIPCONTRA'][i]),
                        xy=(modified2['DIPRADSCON'][i]+ math.radians(90), 1),  # theta, radius
                        xytext=(modified2['DIPRADSCON'][i]+ math.radians(90), 1),    # fraction, fraction
                        textcoords='data',
                        horizontalalignment='left',
                        verticalalignment='bottom', bbox=bbox_props
                        )
        
        for i in range(len(modified2['DIPRADSFAV'].dropna())):    
            ax.annotate(str(-1*modified2['DIPFAVOR'][i]),
                        xy=(math.radians(90) - modified2['DIPRADSFAV'][i], 1),  # theta, radius
                        xytext=(math.radians(90) - modified2['DIPRADSFAV'][i], 1),    # fraction, fraction
                        textcoords='data',
                        horizontalalignment='left',
                        verticalalignment='bottom', bbox=bbox_props
                        )
    
        #### GRAFICAR LEYENDA
        
        coloresnp = np.unique(clasificar_j['TIPO'])
        listacolortipo = []
        listacolortipo.clear()
    
        for i in range(len(coloresnp)):
            filterPRI2 = pd.DataFrame(data=clasificar_j.loc[(clasificar_j['TIPO'] == str(coloresnp[i])), ['TIPO','Color']])
            filterPRI2 = filterPRI2.reset_index(drop=True)
            listacolortipo.append([filterPRI2['TIPO'][0], filterPRI2['Color'][0]])
    
        for i in range(len(coloresnp)):
            r2 = mpatch.Rectangle((0, 0), 0.0, 0.0, color=listacolortipo[i][1], label = str(listacolortipo[i][0]))
            ax.add_patch(r2)
        
        ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))
    
        #### FORMATO GRAFICO EN COORDENADAS POLARES
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_theta_zero_location("S")
        
        draw_figure(window['fig_cv'].TKCanvas, fig)
        #plt.grid()
        #plt.show(block=False) # con este comando al compilar, los graficos funcionan correctamente.

######################### PLOT OPUESTO ##############################
#####################################################################      
        
        
    def plotanisotropiaop():
    
        plt.close(1)
        plt.figure(1)
        fig = plt.gcf()
        DPI = fig.get_dpi()
        ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True, facecolor='#d5de9c')
        fig.set_size_inches(304 * 2 / float(DPI), 404 / float(DPI))
    
        #### GRAFICAR BARRAS

        plt.bar(modified2['DIPRADSFAV'].dropna() + math.radians(90), 1, width=ANCHORADIANS*2, bottom=0.0,
                color=modified2['ColorF'].dropna(), alpha = 0.7)
        plt.bar(math.radians(90) - modified2['DIPRADSCON'].dropna(), 1, width=ANCHORADIANS*2, bottom=0.0,
                color=modified2['ColorC'].dropna(), alpha = 0.7)

        #### GRAFICAR LINEAS

        for i in range(len(modified2['DIPFAVOR'].dropna())):
            theta0 = np.deg2rad([modified2['DIPFAVOR'][i], modified2['DIPFAVOR'][i]]) + math.radians(90)
            R0 = [0,1]
            ax.plot(theta0, R0, lw=2, color = 'black')
        
    
        for i in range(len(modified2['DIPRADSCON'].dropna())):
            theta0 = math.radians(90) - np.deg2rad([modified2['DIPCONTRA'][i], modified2['DIPCONTRA'][i]])
            R0 = [0,1]
            ax.plot(theta0, R0, lw=2, color = 'black')
        
        #### GRAFICAR ANOTACIONES DE ÁNGULOS
    
        bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2)
        
        for i in range(len(modified2['DIPRADSFAV'].dropna())):    
            ax.annotate(str(modified2['DIPFAVOR'][i]),
                        xy=(modified2['DIPRADSFAV'][i]+ math.radians(90), 1),  # theta, radius
                        xytext=(modified2['DIPRADSFAV'][i]+ math.radians(90), 1),    # fraction, fraction
                        textcoords='data',
                        horizontalalignment='left',
                        verticalalignment='bottom', bbox=bbox_props
                        )
        
        for i in range(len(modified2['DIPRADSCON'].dropna())):    
            ax.annotate(str(-1*modified2['DIPCONTRA'][i]),
                        xy=(math.radians(90) - modified2['DIPRADSCON'][i], 1),  # theta, radius
                        xytext=(math.radians(90) - modified2['DIPRADSCON'][i], 1),    # fraction, fraction
                        textcoords='data',
                        horizontalalignment='left',
                        verticalalignment='bottom', bbox=bbox_props
                        )
    
        #### GRAFICAR LEYENDA
        
        coloresnp = np.unique(clasificar_j['TIPO'])
        listacolortipo = []
        listacolortipo.clear()
    
        for i in range(len(coloresnp)):
            filterPRI2 = pd.DataFrame(data=clasificar_j.loc[(clasificar_j['TIPO'] == str(coloresnp[i])), ['TIPO','Color']])
            filterPRI2 = filterPRI2.reset_index(drop=True)
            listacolortipo.append([filterPRI2['TIPO'][0], filterPRI2['Color'][0]])
    
        for i in range(len(coloresnp)):
            r2 = mpatch.Rectangle((0, 0), 0.0, 0.0, color=listacolortipo[i][1], label = str(listacolortipo[i][0]))
            ax.add_patch(r2)
    
        ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))
    
        #### FORMATO GRAFICO EN COORDENADAS POLARES
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_theta_zero_location("S")
        draw_figure(window['fig_cv'].TKCanvas, fig)
        
######################### IMPRIMIR ANGULOS CORRESPONDIENTES ##############################
##########################################################################################
##########################################################################################
        
    def angulos_anisotropia():
        
        global concatdf
        
        data_ordenada = []
        data_ordenada.clear()

        # ordenar en 1 solo dataframe los datos.

        # en contra
        for i in range(len(modified2['DIPCONTRA'].dropna())):
    
            angulomin = modified2['DIPCONTRA'][i] - ANCHOANISO
            angulomax = modified2['DIPCONTRA'][i] + ANCHOANISO

            data_ordenada.append(['CONTRA',modified2['TIPOCONTRA'][i], modified2['DIPCONTRA'][i], angulomin, angulomax])

        # a favor
        for i in range(len(modified2['DIPFAVOR'].dropna())):
    
            angulomin = modified2['DIPFAVOR'][i] - ANCHOANISO
            angulomax = modified2['DIPFAVOR'][i] + ANCHOANISO

            data_ordenada.append(['FAVOR',modified2['TIPOFAV'][i], modified2['DIPFAVOR'][i], -angulomin, -angulomax])
    
        # casos especiales que superan los 90° que estan en contra
        for i in range(len(modified2['DIPCONTRA'].dropna())):
    
            angulomin = modified2['DIPCONTRA'][i] - ANCHOANISO
            angulomax = modified2['DIPCONTRA'][i] + ANCHOANISO
    
            if angulomax > 90:
                angulomax = -(180 - angulomax)
                data_ordenada.append(['FAVOR',modified2['TIPOCONTRA'][i], modified2['DIPCONTRA'][i], -90, angulomax])

        # casos especiales que superan los 90° que estan a favor
        for i in range(len(modified2['DIPFAVOR'].dropna())):
    
            angulomin = modified2['DIPFAVOR'][i] - ANCHOANISO
            angulomax = modified2['DIPFAVOR'][i] + ANCHOANISO
    
            if angulomax > 90:
                angulomax = (180 - angulomax)
                data_ordenada.append(['CONTRA',modified2['TIPOFAV'][i], modified2['DIPFAVOR'][i], angulomax, 90])
    
        onedata = pd.DataFrame(data_ordenada, columns =['TIPO', 'ESTRUCTURA', 'DIP', 'MINDIP', 'MAXDIP'])

        ### revisar que DIPS estan contenidos en otros rangos ###

        resumen_final = []
        resumen_final.clear()

        for i in range(len(onedata)):
            for j in range(len(onedata)):
                a = onedata['MINDIP'][j]
                b = onedata['MAXDIP'][j]
        
                if a < b:
                    g = a
                    h = b
                else:
                    g = b
                    h = a
                if g <  onedata['MINDIP'][i] < h:
                    resumen_final.append([onedata['TIPO'][i], onedata['ESTRUCTURA'][i], onedata['MINDIP'][i],
                                          'Si', onedata['ESTRUCTURA'][j]])
                else:
                    resumen_final.append([onedata['TIPO'][i], onedata['ESTRUCTURA'][i], onedata['MINDIP'][i],
                                          'No', onedata['ESTRUCTURA'][j]])
        for i in range(len(onedata)):
            for j in range(len(onedata)):
                a = onedata['MINDIP'][j]
                b = onedata['MAXDIP'][j]
        
                if a < b:
                    g = a
                    h = b
                else:
                    g = b
                    h = a
                if g <  onedata['MAXDIP'][i] < h:
                    resumen_final.append([onedata['TIPO'][i], onedata['ESTRUCTURA'][i], onedata['MAXDIP'][i],
                                          'Si', onedata['ESTRUCTURA'][j]])
                else:
                    resumen_final.append([onedata['TIPO'][i], onedata['ESTRUCTURA'][i], onedata['MAXDIP'][i],
                                          'No', onedata['ESTRUCTURA'][j]])

        analisis_contencion = pd.DataFrame(resumen_final, columns =['TIPO', 'ESTRUCTURA_1', 'DIP', 'Contenido?', 'ESTRUCTURA_2'])
    
        novan = []
        novan.clear()


        for i in range(len(analisis_contencion)):
            if (analisis_contencion['ESTRUCTURA_1'][i] == 'FALLA') and (analisis_contencion['ESTRUCTURA_2'][i] == 'FALLA') and (analisis_contencion['Contenido?'][i] == 'Si'):
                novan.append(analisis_contencion['DIP'][i])
            elif (analisis_contencion['ESTRUCTURA_1'][i] == 'JOINT') and (analisis_contencion['ESTRUCTURA_2'][i] == 'FALLA') and (analisis_contencion['Contenido?'][i] == 'Si'):
                novan.append(analisis_contencion['DIP'][i])
            elif (analisis_contencion['ESTRUCTURA_1'][i] == 'JOINT') and (analisis_contencion['ESTRUCTURA_2'][i] == 'JOINT') and (analisis_contencion['Contenido?'][i] == 'Si'):
                novan.append(analisis_contencion['DIP'][i])

        final = []
        final.clear()

        for i in range(len(onedata)):
            final.append(onedata['MINDIP'][i])
            final.append(onedata['MAXDIP'][i])
    
        finaldf = pd.DataFrame(final, columns =['ANGULO'])

        df = finaldf[~finaldf['ANGULO'].isin(novan)]
        df = df.sort_values(by=['ANGULO'], inplace=False)
        df = df.drop_duplicates(subset=['ANGULO'])
        df = df[df.ANGULO < 90]
        df = df[df.ANGULO > -90]

        df = df.reset_index(drop=True)
        df.loc[-1] = [-90]
        df.sort_index(inplace=True)
        df = df.reset_index(drop=True)

        addfinal = []
        addfinal.clear()
        addfinal = [90]
        adddf = pd.DataFrame(addfinal, columns =['ANGULO'])
        concatdf = pd.concat([df,adddf])
        concatdf = concatdf.reset_index(drop=True)

######################### SUB PARALELISMO ###########################
##################################################################### 
    def paralelismoplot():
    
        convert = []
        convert.clear()
    
        for i in range(len(clasificar_j['DIPDIR'])):
            convert.append([clasificar_j['TIPO'][i], math.radians(clasificar_j['DIPDIR'][i]), clasificar_j['DIP'][i]/90])
    
        polardf = pd.DataFrame(convert, columns =['TIPO','DIPDIRRAD','CONVDIP'])
    
        # agregar a cada tipo un color diferente
        categories2 = np.unique(polardf['TIPO'])
        colors2 = cm.rainbow(np.linspace(0, 1, len(categories2)))
        colordict2 = dict(zip(categories2, colors2))  
        polardf["Color"] = polardf['TIPO'].apply(lambda x: colordict2[x])
        
        plt.close(1)
        plt.figure(1)
        fig = plt.gcf()
        DPI = fig.get_dpi()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
        ax.scatter(polardf['DIPDIRRAD'], polardf['CONVDIP'], c=polardf['Color'], s=100, cmap='hsv', alpha=0.75)
        fig.set_size_inches(150 * 2 / float(DPI), 200 / float(DPI))
        
        #### GRAFICAR LINEAS

        theta0 = np.deg2rad([DIPDIRTALUD, DIPDIRTALUD]) + math.radians(SUBGRADOS)
        R0 = [0,1]
        ax.plot(theta0, R0, lw=2, color = 'blue')
    
        theta1 = np.deg2rad([DIPDIRTALUD, DIPDIRTALUD]) - math.radians(SUBGRADOS)
        R1 = [0,1]
        ax.plot(theta1, R1, lw=2, color = 'blue')
    
        theta2 = np.deg2rad([DIPDIRTALUD, DIPDIRTALUD]) - math.radians(SUBGRADOS) + math.radians(180)
        R2 = [0,1]
        ax.plot(theta2, R2, lw=2, color = 'blue')

        theta3 = np.deg2rad([DIPDIRTALUD, DIPDIRTALUD]) + math.radians(SUBGRADOS) + math.radians(180)
        R3 = [0,1]
        ax.plot(theta3, R3, lw=2, color = 'blue')
        
        #### GRAFICAR FLECHA
    
        plt.arrow(DIPDIRTALUD/180.*np.pi, 0, 0, 0.8, alpha = 0.5, width = 0.03,
                  edgecolor = 'black', facecolor = 'green', lw = 1, zorder = 5)
        
        
        #### FORMATO GRAFICO EN COORDENADAS POLARES

        ax.set_yticklabels([])
        ax.set_ylim(0,1)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        draw_figure_w_toolbar(window['fig_cv1'].TKCanvas, fig, window['controls_cv'].TKCanvas) # en vez de plotshow
        #plt.grid()
    
    


##################################################################################################################
##################################################################################################################
############################################### PYSIMPLEGUI ######################################################

    fig_dict = {'Left to Right':plotanisotropia, 'Right to Left':plotanisotropiaop}
    listbox_values = list(fig_dict)    
    col_listbox = [[sg.Listbox(values=listbox_values, change_submits=True, size=(18, len(listbox_values)), key='-LISTBOX-')]]

    #, sg.Column(col_listbox)
    
    menu_def = [['File', ['Exit']],
                ['Help', 'About...'],]

    layout = [[sg.Menu(menu_def)],
        [sg.T('Anisotropic Strength Plot', font='Any 18')],
        [sg.T('Type Slope DIPDIR (°): '),
        sg.In(key='inputDIPDIRTALUD', justification='right', size=(8,1), pad=(1,1), do_not_clear=True, default_text='280'), sg.Column(col_listbox)],
        [sg.T('Type Sub-Parallelism Angle (°): '),
        sg.In(key='inputSUBDEGREES', justification='right', size=(8,1), pad=(1,1), do_not_clear=True, default_text='30')],
        [sg.B('Plot'), sg.B('Exit')],
        [sg.T('Controls:')],
        [sg.Canvas(key='controls_cv')],
        [sg.T('Figure:')],
        [sg.Column(
            layout=[[sg.Canvas(key='fig_cv1', size=(150 * 2, 400)),
                sg.Canvas(key='fig_cv', size=(300 * 2, 400))]
                ],
            background_color='#DAE0E6',
            pad=(0, 0)
            )],
        [sg.B('Angles')]

    ]

    window = sg.Window('SRK Anisotropic Function', layout)

    while True:
        event, values = window.read()
        print(event, values)
        if event in (None, 'Exit'):  # always,  always give a way out!
            break
        
        
        elif event == 'Plot':
            
            try:
            
                choice = values['-LISTBOX-'][0]
                func = fig_dict[choice]
                # ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE
                DIPDIRTALUD = float(values['inputDIPDIRTALUD'])
                SUBGRADOS = float(values['inputSUBDEGREES']) 
                subparalelismo()
                angulos_anisotropia()
            
                if func == plotanisotropia:
                    try:
                        func()
                    except:
                        sg.PopupError('Error')
                    
                elif func == plotanisotropiaop:
                    try:
                        func()
                        
                        lista_reverse = []
                        lista_reverse.clear()
                        k = len(concatdf) - 1

                        while k != -1:
                            lista_reverse.append(-concatdf['ANGULO'][k])
                            k -= 1

                        reversedf = pd.DataFrame(lista_reverse, columns =['ANGULO'])
                        print(reversedf)
                        #concatdf = reversedf
                        
                    except:
                        sg.PopupError('Error')
                    
                paralelismoplot()
                sg.Print(str(newj))
                
                if func == plotanisotropiaop:
                    sg.Print(str(reversedf))
                else:
                    sg.Print(str(concatdf))
                
            except:
                
                sg.PopupError('Error')
                
        if event == 'About...':
            sg.Popup('Developed by Jorge Alvarez C.')     

    window.close()

table_example()

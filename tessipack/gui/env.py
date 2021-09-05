
class Environment(object):
    ''' This is Class is for to share all the object along all the files

    Tabels,figures and variable will be assiagned here.

    '''
    #---------------------------Light Curve Param----------------------------- #

    default_cluster="star"
    sigma=5
    plot_nearby=-1 #set this to 1 and -1 to toggle plotting of nearby source
    draw_nearby_names=1

    selection_program = None
    text_custom_star_ra=None
    text_custom_star_dec=None
    text_custom_star_sector=None
    custom_star_download_button=None
    selection_program_text=None



    text_banner=None
    Message=None

    v_flag_duplicate=None
    v_flag_source=None
    v_flag_check=None


    text_flag_duplicate=None
    text_flag_source=None
    text_flag_check=None

    text_catalog_query=None
    text_id_mycatalog_query=None
    text_id_query=None
    text_Notes=None
    text_Notes_w=None

    show_error_bar=None

    aperture_setting=1 #1 current and -1 default

    # extra_flag_file='/home/dinilbose/PycharmProjects/light_cluster/cluster/Collinder_69/Data/extra_flag.flag'
    extra_flag_file='extra_flag.flag'

    current_flux_dataframe=None
    #----------------------------Source Table--------------------------------- #

    tb_source=2
    tb_lightcurve=None
    tb_periodogram=None
    tb_tpf=None
    tpf_flux=None
    table_periodogram=None
    tb_periodogram_se_tb=None
    tb_nearby=None
    tb_catalog_main=None
    tb_catalog_all=None
    tb_isochrone=None
    tb_nearby_star=None
    tb_nearby_star_table=None

    fig_hr=None
    fig_position=None

    text_age=None
    text_metallicity=None
    text_extinction_av=None
    text_distance=None
    generate_isochrone_button=None
    delete_isochrone_button=None
    int_select_sector=None
    sector=6
    # ----------------------------Figures ------------------------------------ #
    plot_width = 810
    plot_height = 300
    fiducial_frame=0
    selection={'selection_color':"red",'nonselection_fill_alpha':0.7,
    'nonselection_fill_color': "#1F77B4",
    'nonselection_line_color': "#1F77B4",
    'nonselection_line_alpha':0.7}

    selection_osc={'line_color':"green",'nonselection_line_color': "#1F77B4",'nonselection_line_alpha':1}



    selection_l={'line_color':"#1F77B4",'nonselection_line_color': "#1F77B4",'nonselection_line_alpha':1}

    selection_2={'selection_color':"red",
    'nonselection_fill_color': "blue",
    'nonselection_line_color': "blue",
    'nonselection_line_alpha':0.7}

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ]

    fig_lightcurve=None
    fig_tpf=None
    fig_stretch=None
    fig_periodogram=None
    # --------------------------- Button -------------------------------------#

    Generate_lc_button=None
    reset_axis_lc_button=None
    reset_dflt_lc_button=None

    next_button=None
    previous_button=None

    reset_axes_prd_button=None
    saveas_prd_tb_button=None
    save_prd_tb_button=None
    reset_prd_tb_button=None
    load_prd_tb_button=None
    save_userinput_button=None
    save_current_button=None
    aperture_selection_button=None
    update_catalog_button=None
    update_id_mycatalog_button=None
    update_id_button=None

    div_spinner=None
    show_spinner_button=None

    spinner_text = """
    <!-- https://www.w3schools.com/howto/howto_css_loader.asp -->
    <div class="loader">
    <style scoped>
    .loader {
        border: 16px solid #f3f3f3; /* Light grey */
        border-top: 16px solid #3498db; /* Blue */
        border-radius: 50%;
        width: 120px;
        height: 120px;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    </div>
    """

    from bokeh.models.widgets import  Div
    div_spinner = Div(text="",width=120,height=120)







#  Analysis Window

    tb_lc_an1=None
    tb_lc_an2=None
    fig_lc_an1=None
    fig_lc_an2=None
    fig_pr_an1= None
    fig_pr_an2=None

    tb_pr_an1=None
    tb_pr_an2=None
    fig_pr_an1= None
    fig_pr_an2= None

    text_p1= None
    text_p0= None
    text_p2= None
    text_pe= None


    period_recompute_button=None
    fold_button=None
    check_analysis=None










# Control Panel

    Samp_status=0
    Samp_selection='Server'
    Control_function='None'
    catalog_main=''
    catalog_whole_gaia=None
    ds9_command_button=None
    ds9_catalog_status=None
    gaia_Gmag_start=None
    gaia_Gmag_end=None
    whole_gaia_filter=None

    gaia_Gmag_start_text=None
    gaia_Gmag_end_text=None
    gaia_update_button=None
    gaia_radius_text=None
    isochrone_data=None
    catalog_find_from_isocrhone=None

    text_banner_Gmag= None
    text_banner_bp_rp= None
    text_banner_dmin=None


    #interactive

    fig_tpfint=None
    stretch_sliderint=None
    dnu_slider=None
    r_button=None
    l_button=None
    rr_button=None
    ll_button=None

    minimum_frequency=None
    maximum_frequency=None
    maxdnu=None

    minimum_frequency_text=None
    maximum_frequency_text=None
    maxdnu_text=None
    update_int_button=None

    mesa_osc_data=None
    fig_mesa_int=None
    mesa_int_slider=None
    tb_mesa_osc_l0=None
    tb_mesa_osc_l1=None
    tb_mesa_osc_l2=None

    int_select_mass=None
    int_select_y=None
    int_select_age=None
    int_select_z=None
    int_select_alpha=None

    comp_data=None

    interactive_file_control=-1

    source_int_slider=None
    frequency_minimum_text1=None
    frequency_maximum_text1=None
    frequency_maxdnu_text1=None
    update_int_source_button=None


    set_value_dict=None


    tb_oscillation_modell0=None
    tb_oscillation_modell1=None
    tb_oscillation_modell2=None
    plot_mesa_osc=None
    text_osc_query=None
    # update_all=None

    # from bokeh.models.widgets import  Div
    # self.env.div_spinner = Div(text="",width=120,height=120)
    # def show_spinner():
    #     self.env.div_spinner.text = self.env.spinner_text
    # def hide_spinner():
    #     self.env.div_spinner.text = ""
    # self.env.show_spinner_button = Button(label='Show Spinner', width=100)
    # self.env.show_spinner_button.on_click(show_spinner)


    # doc = None                      # pointer to curdoc()
    # ly = None                       # main bokeh layout
    #
    # bridge_row = None               # messages bridge row to add to the layout
    # tabs = None                     # tabs structure
    # cur_plotted_cols = []           # Current columns list used in all the plots
    #
    # sidebar = None                  # sidebar, actually it is a column
    # tabs_widget = None              # tabs widget
    # flagger_select = None           # flag selection widget
    # wmts_map = None                 # tile server map
    # wmts_map_scatter = None         # scatter for the tile server map
    # flags_control_col = None        # flags control column (visibility and flag updates)
    # show_titles = False             # Whether show titles on plots or not

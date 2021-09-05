import sys
from bokeh.layouts import row,column
from bokeh.layouts import grid
from bokeh.plotting import curdoc, figure,show
from bokeh.models import Panel
import pandas
from bokeh.models import Button  # for saving data

#import catalog
#import env
#import aperture
#from importlib import reload
from bokeh.models.widgets import Tabs
from bokeh.layouts import layout, Spacer


#reload(env)
#reload(catalog)
#reload(aperture)
from catalog import Catalog
from env import Environment
from aperture import Aperture
from periodogram import Periodogram
from control_panel import Control_panel
from lcanalysis import Lcanalysis
#from interactive2 import Interactive

env=Environment

# Class run(Environment):

Environment()
Catalog()
Aperture()
Periodogram()
#Lcanalysis()
#Interactive()
control=Control_panel()


col1=row(column(env.fig_tpf,env.fig_stretch),column(env.show_error_bar,env.fig_lightcurve,env.cadence_slider,env.fig_periodogram),column(env.table_periodogram,env.tb_nearby_star_table))
# col1=row(column(env.fig_lightcurve))
# default_lightcurve,nextbutton,previousbutton
col2=row(env.previous_button,env.Generate_lc_button,env.reset_axis_lc_button,env.reset_dflt_lc_button,env.next_button,env.aperture_selection_button,env.save_current_button,env.toggle_button)
# ,env.show_spinner_button,env.div_spinner
col4=row(env.text_flag_duplicate,env.text_flag_source,env.text_flag_check,env.save_userinput_button,env.show_error_bar)
col3=row(env.text_banner)
col5=row(column(env.text_banner_bp_rp,env.text_banner_Gmag,env.text_banner_dmin))
col6=row(column(env.text_Notes_w))

layout=column(col1,col2,col3,col4,col5,col6)
tab0= Panel(child=layout, title = 'Aperture')
layout2=column(env.fig_periodogram)

nxt_prv_button=row(env.next_button,env.previous_button,sizing_mode='fixed')
table_button=row(env.reset_axes_prd_button,env.reset_prd_tb_button,sizing_mode='fixed')
# widgets = column(row(axesbutton,reset_se_button),nxt_prv_button,savebutton)
widgets = column(table_button,nxt_prv_button,env.load_prd_tb_button,env.saveas_prd_tb_button,env.save_prd_tb_button,sizing_mode='fixed')

# main_row = row(fig_periodogram,fig_periodogram1,env.table_periodogram)
main_row = row(env.fig_periodogram,env.fig_periodogram1,env.table_periodogram)
series = row(env.fig_periodogram2, env.fig_periodogram3,widgets,sizing_mode='fixed')
last_row = row(env.fig_periodogram4,env.fig_lightcurve)
layout = column(main_row, series,last_row)
tab1 = Panel(child=layout, title = 'Periodogram')

# tab1= Panel(child=layout2, title = 'Periodogram')


#
flag_lay1=row(env.text_flag_duplicate,env.text_flag_source,env.text_flag_check,env.save_userinput_button)
next_lay1=row(env.previous_button,env.next_button)
dflt_lay1=row(env.reset_axis_lc_button,env.reset_dflt_lc_button)
next_dflt_lay=column(next_lay1,dflt_lay1)
query_lay1=column(row(env.text_cluster_query,env.update_cluster_button),row(env.text_catalog_query,env.update_catalog_button),row(env.int_select_sector),row(env.text_id_mycatalog_query,env.update_id_mycatalog_button),row(env.text_id_query,env.update_id_button))
# iso_lay1=column(env.text_age,env.text_metallicity,env.text_extinction_av,env.text_distance,row(env.generate_isochrone_button,env.delete_isochrone_button))
iso_lay1=column(env.selection_program,row(env.text_custom_star_ra),row(env.text_custom_star_dec),row(env.text_custom_star_sector,env.custom_star_download_button))

text_lay1=column(env.text_banner_bp_rp,env.text_banner_Gmag,env.text_banner_dmin)
layer_1=column(query_lay1,next_dflt_lay,iso_lay1)
notes_lay1=column(env.text_Notes_w)

# layout_catalog=row(column(env.text_catalog_query,env.text_id_mycatalog_query,env.text_id_query),column(env.update_catalog_button,env.update_id_mycatalog_button,env.update_id_button),env.fig_hr)
layout_catalog=column(row(layer_1,env.fig_hr,column(env.fig_lightcurve,env.fig_periodogram)),row(env.text_banner),flag_lay1,text_lay1,notes_lay1)
tab_2 = Panel(child=layout_catalog, title = 'Catalog')



# layout_plots=column(row(layer_1,env.fig_position,column(env.fig_lightcurve,env.fig_periodogram)),row(env.text_banner),flag_lay1,text_lay1,notes_lay1)
# tab_plots = Panel(child=layout_plots, title = 'Position')
#
#
# lcanalysis_layout=column(row(env.fig_lc_an1,env.fig_pr_an1,env.table_periodogram),
#                          row(env.fold_select_1,env.flux_name_1,env.fold_button,env.period_recompute_button,
#                              env.text_p0,env.text_p1,env.text_p2,env.text_pe),
#                          row(env.fig_lc_an2,env.fig_pr_an2),
#                          row(env.fold_select_2,env.flux_name_2,env.fold_button2,env.period_recompute_button2,
#                              env.text2_p0,env.text2_p1,env.text2_p2,env.text2_pe),)
# lcanalysis_panel=Panel(child=lcanalysis_layout, title = 'Analysis')
#
#
#
# tab_control_panel_layout=column(row(env.Samp_status,env.Samp_selection),
#                                 row(env.gaia_Gmag_start_text,env.gaia_Gmag_end_text,
#                                     env.gaia_update_button),
#                                 row(column(env.ds9_status,env.ds9_catalog_status),env.gaia_radius_text,env.ds9_command_button),env.check_analysis)
# tab_control_panel=Panel(child=tab_control_panel_layout, title = 'Control Panel')
#
#
#
# # tab_interactive_layout = layout([env.fig_tpfint, [Spacer(height=20), env.stretch_sliderint]],
# #                              [env.ll_button, Spacer(width=30), env.l_button,
# #                               Spacer(width=25), env.dnu_slider, Spacer(width=30),
# #                               env.r_button, Spacer(width=23), env.rr_button])
# int_selection=column(env.int_select_mass,env.int_select_z,env.int_select_y,env.int_select_age,env.int_select_alpha)
# source_section=column(env.mesa_int_slider,env.source_int_slider,
#                       env.frequency_minimum_text1,env.frequency_maximum_text1,
#                       env.update_int_source_button,env.update_int_reload_button )
#
# tab_int_1=column(row(env.fig_tpfint,env.stretch_sliderint),
#                               row(env.ll_button,env.l_button,env.dnu_slider,env.r_button,env.rr_button),
#                               row(env.frequency_minimum_text,env.frequency_maximum_text,env.frequency_maxdnu_text),
#                               row(env.update_int_button,nxt_prv_button),
#                               row(source_section,int_selection))
#
# tab_interactive_layout=column(row(tab_int_1,column(env.fig_periodogram,env.fig_mesa_int),env.table_periodogram),row(env.text_osc_query,env.plot_mesa_osc),row(env.text_banner,))
# tab_c=Panel(child=tab_interactive_layout, title = 'Interactive')
#




# tab2=[]
#tabs = Tabs(tabs = [tab_2,tab0,tab1,tab_plots,lcanalysis_panel,tab_control_panel,tab_c])
tabs = Tabs(tabs = [tab_2,tab0])

# tabs = Tabs(tabs = [tab_2,tab0,tab1,tab_plots,lcanalysis_panel,tab_control_panel])

#tabs = Tabs(tabs = [tab_2,tab0,tab1,tab_plots,lcanalysis_panel,tab_control_panel,tab_interactive_layout])





curdoc().add_periodic_callback(env.Control_function, 1000)
curdoc().add_root(tabs)
#curdoc()

from bokeh.client import push_session
# session = push_session(curdoc())
#script = autoload_server(session_id=session.id)
#session.loop_until_closed()

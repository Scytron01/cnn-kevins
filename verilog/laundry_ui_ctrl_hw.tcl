package require qsys
set_module_property NAME laundry_ui_ctrl
set_module_property DISPLAY_NAME "Laundry User Buttons and Switches"
set_module_property VERSION 1.0
set_module_property GROUP "Laundry Sorter"
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL laundry_ui_ctrl
add_fileset_file laundry_ui_ctrl.v VERILOG PATH laundry_ui_ctrl.v TOP_LEVEL_FILE
add_interface clock clock end
add_interface_port clock clk clk Input 1
add_interface reset reset end
set_interface_property reset associatedClock clock
add_interface_port reset reset_n reset_n Input 1
add_interface conduit conduit end
add_interface_port conduit key_n export Input 4
add_interface_port conduit switches export Input 10
add_interface avs avalon end
set_interface_property avs associatedClock clock
set_interface_property avs associatedReset reset
set_interface_property avs addressUnits WORDS
set_interface_property avs readLatency 0
add_interface_port avs avs_address address Input 2
add_interface_port avs avs_read read Input 1
add_interface_port avs avs_readdata readdata Output 32
add_interface_port avs avs_write write Input 1
add_interface_port avs avs_writedata writedata Input 32
add_interface_port avs avs_waitrequest waitrequest Output 1

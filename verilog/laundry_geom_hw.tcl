# Platform Designer component descriptor; verify against installed Quartus release.
package require qsys
set_module_property NAME laundry_geom_ip
set_module_property DISPLAY_NAME "Laundry Geometry Extractor"
set_module_property VERSION 1.0
set_module_property GROUP "Laundry Sorter"
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL laundry_geom_ip
add_fileset_file laundry_geom_ip.v VERILOG PATH laundry_geom_ip.v TOP_LEVEL_FILE

add_interface clock clock end
add_interface_port clock clk clk Input 1
add_interface reset reset end
set_interface_property reset associatedClock clock
add_interface_port reset reset_n reset_n Input 1

add_interface pixel_sink avalon_streaming end
set_interface_property pixel_sink associatedClock clock
set_interface_property pixel_sink associatedReset reset
set_interface_property pixel_sink dataBitsPerSymbol 1
set_interface_property pixel_sink symbolsPerBeat 1
add_interface_port pixel_sink sink_valid valid Input 1
add_interface_port pixel_sink sink_data data Input 1
add_interface_port pixel_sink sink_sop startofpacket Input 1
add_interface_port pixel_sink sink_eop endofpacket Input 1
add_interface_port pixel_sink sink_ready ready Output 1

add_interface csr avalon end
set_interface_property csr associatedClock clock
set_interface_property csr associatedReset reset
set_interface_property csr addressUnits WORDS
add_interface_port csr csr_address address Input 3
add_interface_port csr csr_read read Input 1
add_interface_port csr csr_readdata readdata Output 32

add_interface irq interrupt end
set_interface_property irq associatedAddressablePoint csr
set_interface_property irq associatedClock clock
set_interface_property irq associatedReset reset
add_interface_port irq irq irq Output 1

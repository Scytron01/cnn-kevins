package require qsys
set_module_property NAME vga_fb_dma
set_module_property DISPLAY_NAME "Laundry VGA 640x480 RGB565 Scanout"
set_module_property VERSION 1.0
set_module_property GROUP "Laundry Sorter"
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL vga_fb_dma
add_fileset_file vga_fb_dma.v VERILOG PATH vga_fb_dma.v TOP_LEVEL_FILE
add_fileset_file vga_timing_640x480.v VERILOG PATH vga_timing_640x480.v
add_interface clock clock end
add_interface_port clock clk_50 clk Input 1
add_interface reset reset end
set_interface_property reset associatedClock clock
add_interface_port reset reset_n reset_n Input 1
add_interface fb_master avalon start
set_interface_property fb_master associatedClock clock
set_interface_property fb_master associatedReset reset
set_interface_property fb_master addressUnits WORDS
set_interface_property fb_master burstcountUnits WORDS
set_interface_property fb_master maximumPendingReadTransactions 1
set_interface_property fb_master readLatency 0
add_interface_port fb_master avm_address address Output 32
add_interface_port fb_master avm_read read Output 1
add_interface_port fb_master avm_burstcount burstcount Output 9
add_interface_port fb_master avm_readdata readdata Input 32
add_interface_port fb_master avm_readdatavalid readdatavalid Input 1
add_interface_port fb_master avm_waitrequest waitrequest Input 1
add_interface conduit conduit end
add_interface_port conduit vga_clk export Output 1
add_interface_port conduit vga_hsync_n export Output 1
add_interface_port conduit vga_vsync_n export Output 1
add_interface_port conduit vga_blank_n export Output 1
add_interface_port conduit vga_sync_n export Output 1
add_interface_port conduit vga_r export Output 8
add_interface_port conduit vga_g export Output 8
add_interface_port conduit vga_b export Output 8

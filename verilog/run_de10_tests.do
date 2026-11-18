transcript on
if {[file exists work_de10]} { vdel -lib work_de10 -all }
vlib work_de10
vlog -2001 -work work_de10 verilog/cnn_mac_unit.v
vlog -2001 -work work_de10 verilog/cnn_top.v
vlog -2001 -work work_de10 verilog/cnn_avalon_slave.v
vlog -2001 -work work_de10 verilog/tb_cnn_avalon_slave.v
vlog -2001 -work work_de10 verilog/laundry_ui_ctrl.v
vlog -2001 -work work_de10 verilog/tb_laundry_ui_ctrl.v
vlog -2001 -work work_de10 verilog/vga_timing_640x480.v
vlog -2001 -work work_de10 verilog/tb_vga_timing_640x480.v
vsim -voptargs=+acc work_de10.tb_laundry_ui_ctrl
run -all
quit -sim
vsim -voptargs=+acc work_de10.tb_vga_timing_640x480
run -all
quit -sim
vsim -voptargs=+acc work_de10.tb_cnn_avalon_slave
run -all
quit -f

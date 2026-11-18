transcript on
if {[file exists work]} {
    vdel -lib work -all
}
vlib work
vlog -2001 -work work laundry_geom_ip.v
vlog -2001 -work work tb_laundry_geom.v
vsim -voptargs=+acc work.tb_laundry_geom
run -all
quit -f

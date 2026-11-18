`timescale 1ns/1ps
module tb_vga_timing_640x480;
    reg clk_50=0,reset_n=0,run=0;
    wire vga_clk,pixel_tick,active_video,line_start,frame_start,hsync_n,vsync_n;
    wire [9:0] pixel_x,pixel_y;
    vga_timing_640x480 dut(.clk_50(clk_50),.reset_n(reset_n),.run(run),.vga_clk(vga_clk),
        .pixel_tick(pixel_tick),.pixel_x(pixel_x),.pixel_y(pixel_y),.active_video(active_video),
        .line_start(line_start),.frame_start(frame_start),.hsync_n(hsync_n),.vsync_n(vsync_n));
    always #10 clk_50=~clk_50;
    initial begin
        repeat(3) @(negedge clk_50); reset_n=1; run=1;
        repeat(801) @(posedge vga_clk); #1;
        if(pixel_x!==0 || pixel_y!==1) $fatal(1,"Expected x=0,y=1 after 800 pixel ticks");
        if(active_video!==1) $fatal(1,"Line 1 start should be active");
        repeat(657) @(posedge vga_clk); #1;
        if(hsync_n!==0) $fatal(1,"HSYNC should be active low at x=656");
        repeat(97) @(posedge vga_clk); #1;
        if(hsync_n!==1) $fatal(1,"HSYNC should be inactive after 96 pixels");
        $display("PASS: VGA_640x480_TIMING"); $finish;
    end
endmodule

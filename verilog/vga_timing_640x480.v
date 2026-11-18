// 640x480 VGA timing from a 50 MHz input clock.
// The nominal 25 MHz pixel clock is divided by two. Pixel counters/scanout update
// on the falling edge so RGB/sync/blank are stable before the next VGA rising edge.
module vga_timing_640x480(
    input wire clk_50,input wire reset_n,input wire run,
    output reg vga_clk,output wire pixel_tick,
    output reg [9:0] pixel_x,output reg [9:0] pixel_y,
    output wire active_video,output wire line_start,output wire frame_start,
    output wire hsync_n,output wire vsync_n
);
    localparam H_VISIBLE=640,H_FRONT=16,H_SYNC=96,H_BACK=48,H_TOTAL=800;
    localparam V_VISIBLE=480,V_FRONT=10,V_SYNC=2,V_BACK=33,V_TOTAL=525;
    assign pixel_tick=run && (vga_clk==1'b1);
    assign active_video=run && (pixel_x<H_VISIBLE) && (pixel_y<V_VISIBLE);
    assign line_start=pixel_tick && (pixel_x==0);
    assign frame_start=pixel_tick && (pixel_x==0) && (pixel_y==0);
    assign hsync_n=!((pixel_x>=H_VISIBLE+H_FRONT)&&(pixel_x<H_VISIBLE+H_FRONT+H_SYNC));
    assign vsync_n=!((pixel_y>=V_VISIBLE+V_FRONT)&&(pixel_y<V_VISIBLE+V_FRONT+V_SYNC));
    always @(posedge clk_50 or negedge reset_n) begin
        if(!reset_n) begin vga_clk<=0; pixel_x<=0; pixel_y<=0; end
        else begin
            vga_clk<=~vga_clk;
            if(pixel_tick) begin
                if(pixel_x==H_TOTAL-1) begin
                    pixel_x<=0;
                    if(pixel_y==V_TOTAL-1) pixel_y<=0;
                    else pixel_y<=pixel_y+1'b1;
                end else pixel_x<=pixel_x+1'b1;
            end
        end
    end
endmodule

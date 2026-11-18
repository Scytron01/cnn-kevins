// RGB565 VGA framebuffer scanout. A 32-bit Avalon read returns two pixels.
// FRAMEBUFFER_BASE_WORD is an Avalon word address in FPGA SDRAM, NOT a HPS CPU address.
module vga_fb_dma #(
    parameter [31:0] FRAMEBUFFER_BASE_WORD=32'd0
)(
    input wire clk_50,input wire reset_n,
    input wire avm_waitrequest,input wire avm_readdatavalid,input wire [31:0] avm_readdata,
    output reg avm_read,output reg [31:0] avm_address,output reg [8:0] avm_burstcount,
    output wire vga_clk,output reg vga_hsync_n,output reg vga_vsync_n,
    output reg vga_blank_n,output wire vga_sync_n,
    output reg [7:0] vga_r,output reg [7:0] vga_g,output reg [7:0] vga_b,
    output reg underflow
);
    localparam [1:0] DMA_IDLE=0,DMA_REQ=1,DMA_DATA=2;
    reg [1:0] dma_state;
    reg [8:0] fetch_line,pending_line,read_word_count;
    reg pending_fetch;
    reg [1:0] line_ready;
    reg frame_ready;
    reg [15:0] line_buffer_0 [0:639];
    reg [15:0] line_buffer_1 [0:639];
    wire pixel_tick;
    wire [9:0] pixel_x,pixel_y;
    wire active_video,line_start,frame_start;
    wire timing_hsync_n,timing_vsync_n;
    wire [15:0] current_pixel=(pixel_x<640)?(pixel_y[0]?line_buffer_1[pixel_x]:line_buffer_0[pixel_x]):16'd0;
    wire current_line_ready=pixel_y[0]?line_ready[1]:line_ready[0];

    vga_timing_640x480 u_timing(
        .clk_50(clk_50),.reset_n(reset_n),.run(frame_ready),.vga_clk(vga_clk),
        .pixel_tick(pixel_tick),.pixel_x(pixel_x),.pixel_y(pixel_y),
        .active_video(active_video),.line_start(line_start),.frame_start(frame_start),
        .hsync_n(timing_hsync_n),.vsync_n(timing_vsync_n)
    );
    assign vga_sync_n=1'b0; // separate H/V sync mode for the DE10-Standard VGA DAC
    always @* begin
        avm_read=(dma_state==DMA_REQ);
        avm_address=FRAMEBUFFER_BASE_WORD+(fetch_line*32'd320)+read_word_count;
        avm_burstcount=9'd1;
    end
    always @(posedge clk_50 or negedge reset_n) begin
        if(!reset_n) begin
            dma_state<=DMA_IDLE; fetch_line<=0; pending_line<=0; pending_fetch<=1;
            read_word_count<=0; line_ready<=0; frame_ready<=0; underflow<=0;
            vga_r<=0; vga_g<=0; vga_b<=0; vga_hsync_n<=1; vga_vsync_n<=1; vga_blank_n<=0;
        end else begin
            if(line_start && frame_ready) begin
                if(pixel_y==479) pending_line<=0;
                else pending_line<=pixel_y+1'b1;
                pending_fetch<=1;
                if(pixel_y!=0) line_ready[~pixel_y[0]]<=0;
                if(pixel_y!=0 && !line_ready[pixel_y[0]]) underflow<=1;
            end
            case(dma_state)
                DMA_IDLE: if(pending_fetch) begin
                    fetch_line<=pending_line; read_word_count<=0; pending_fetch<=0; dma_state<=DMA_REQ;
                end
                DMA_REQ: if(!avm_waitrequest) begin
                    dma_state<=DMA_DATA;
                end
                DMA_DATA: if(avm_readdatavalid) begin
                    if(!fetch_line[0]) begin
                        line_buffer_0[read_word_count*2]<=avm_readdata[15:0];
                        line_buffer_0[read_word_count*2+1]<=avm_readdata[31:16];
                    end else begin
                        line_buffer_1[read_word_count*2]<=avm_readdata[15:0];
                        line_buffer_1[read_word_count*2+1]<=avm_readdata[31:16];
                    end
                    if(read_word_count==319) begin
                        line_ready[fetch_line[0]]<=1;
                        if(fetch_line==0) frame_ready<=1;
                        dma_state<=DMA_IDLE;
                    end else begin
                        read_word_count<=read_word_count+1'b1;
                        dma_state<=DMA_REQ;
                    end
                end
                default: dma_state<=DMA_IDLE;
            endcase
            if(pixel_tick) begin
                vga_hsync_n<=timing_hsync_n;
                vga_vsync_n<=timing_vsync_n;
                vga_blank_n<=active_video && current_line_ready;
                if(active_video && current_line_ready) begin
                    vga_r<={current_pixel[15:11],current_pixel[15:13]};
                    vga_g<={current_pixel[10:5],current_pixel[10:9]};
                    vga_b<={current_pixel[4:0],current_pixel[4:2]};
                end else begin vga_r<=0; vga_g<=0; vga_b<=0; end
            end
        end
    end
endmodule

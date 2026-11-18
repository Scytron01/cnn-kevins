// Streaming 64x64 binary-mask geometry extractor for Intel Platform Designer.
module laundry_geom_ip #(
    parameter IMG_W=64, IMG_H=64
)(
    input wire clk, reset_n,
    input wire sink_valid, sink_data, sink_sop, sink_eop,
    output wire sink_ready,
    input wire [2:0] csr_address,
    input wire csr_read,
    output reg [31:0] csr_readdata,
    output wire irq
);
    localparam XW=$clog2(IMG_W), YW=$clog2(IMG_H);
    reg [XW-1:0] x_count;
    reg [YW-1:0] y_count;
    reg [12:0] area_count;
    reg [7:0] xmin,xmax,ymin,ymax;
    reg previous_pixel;
    reg [6:0] row_run_count;
    reg [6:0] row_runs [0:IMG_H-1];
    reg finalizing, frame_done;
    reg [YW-1:0] final_row;
    reg [7:0] two_run_rows;
    wire take_pixel=sink_valid && sink_ready;
    wire [6:0] run_next=row_run_count+((sink_data && !previous_pixel)?7'd1:7'd0);
    wire [7:0] bbox_w=(xmax>=xmin)?(xmax-xmin+1'b1):8'd0;
    wire [7:0] bbox_h=(ymax>=ymin)?(ymax-ymin+1'b1):8'd0;
    wire [15:0] h9=bbox_h*16'd9, h14=bbox_h*16'd14;
    wire [7:0] lower_start=ymin+(h9>>4);
    wire [7:0] lower_end=ymin+(h14>>4);
    assign sink_ready=!finalizing;
    assign irq=frame_done;
    integer i;
    always @(posedge clk or negedge reset_n) begin
      if(!reset_n) begin
        x_count<=0; y_count<=0; area_count<=0;
        xmin<=63; xmax<=0; ymin<=63; ymax<=0;
        previous_pixel<=0; row_run_count<=0;
        finalizing<=0; frame_done<=0; final_row<=0; two_run_rows<=0;
        for(i=0;i<IMG_H;i=i+1) row_runs[i]<=0;
      end else if(finalizing) begin
        if(bbox_h!=0 && final_row>=lower_start && final_row<=lower_end && row_runs[final_row]==2)
          two_run_rows<=two_run_rows+1'b1;
        if(final_row==IMG_H-1) begin finalizing<=0; frame_done<=1; end
        else final_row<=final_row+1'b1;
      end else if(take_pixel) begin
        if(sink_sop) begin
          x_count<=1; y_count<=0; area_count<=sink_data?1:0;
          xmin<=sink_data?0:63; xmax<=0; ymin<=sink_data?0:63; ymax<=0;
          previous_pixel<=sink_data; row_run_count<=sink_data?1:0;
          frame_done<=0; two_run_rows<=0;
        end else begin
          if(sink_data) begin
            area_count<=area_count+1'b1;
            if(x_count<xmin) xmin<=x_count;
            if(x_count>xmax) xmax<=x_count;
            if(y_count<ymin) ymin<=y_count;
            if(y_count>ymax) ymax<=y_count;
          end
          if(x_count==IMG_W-1) begin
            row_runs[y_count]<=run_next; row_run_count<=0; previous_pixel<=0; x_count<=0;
            if(y_count==IMG_H-1) y_count<=0; else y_count<=y_count+1'b1;
          end else begin
            x_count<=x_count+1'b1; row_run_count<=run_next; previous_pixel<=sink_data;
          end
          if(sink_eop) begin finalizing<=1; final_row<=0; end
        end
      end
    end
    always @(*) begin
      csr_readdata=0;
      if(csr_read) case(csr_address)
        0: csr_readdata={29'd0,frame_done,finalizing,1'b0};
        1: csr_readdata={19'd0,area_count};
        2: csr_readdata={ymax,ymin,xmax,xmin};
        3: csr_readdata={24'd0,two_run_rows};
        4: csr_readdata={16'd0,bbox_h,bbox_w};
        default: csr_readdata=0;
      endcase
    end
endmodule

module cnn_top (
    input wire clk, input wire reset_n, input wire start,
    input wire [31:0] geometry_q7,
    input wire in_valid, output wire in_ready, input wire in_pixel,
    output reg result_valid, output reg [2:0] result_class, output wire busy
);
    localparam [4:0] S_IDLE=0, S_LOAD=1, S_LAYER_SETUP=2, S_MAC=3,
        S_CONV_FINISH=4, S_GAP_SETUP=5, S_GAP_MAC=6,
        S_FC_SETUP=7, S_FC_MAC=8, S_FC_FINISH=9, S_ARGMAX=10;
    localparam integer Q24_SHIFT=24;
    reg [4:0] state;
    reg [12:0] load_count;
    reg [31:0] geometry_latched;
    reg input_mem [0:4095];
    reg signed [7:0] act1_mem [0:8191];
    reg signed [7:0] act2_mem [0:4095];
    reg signed [7:0] act3_mem [0:2047];
    reg signed [7:0] fc_input_mem [0:35];
    reg signed [7:0] fc1_act_mem [0:63];
    reg signed [31:0] logits_mem [0:4];
    reg signed [7:0] conv1_w [0:71];
    reg signed [31:0] conv1_b [0:7];
    reg [31:0] conv1_mul_mem [0:0];
    reg signed [7:0] conv2_w [0:1151];
    reg signed [31:0] conv2_b [0:15];
    reg [31:0] conv2_mul_mem [0:0];
    reg signed [7:0] conv3_w [0:4607];
    reg signed [31:0] conv3_b [0:31];
    reg [31:0] conv3_mul_mem [0:0];
    reg signed [7:0] fc1_w [0:2303];
    reg signed [31:0] fc1_b [0:63];
    reg [31:0] fc1_mul_mem [0:0];
    reg signed [7:0] fc2_w [0:319];
    reg signed [31:0] fc2_b [0:4];
    reg [31:0] gap_mul_mem [0:0];
    reg [31:0] geom_mul_mem [0:0];

    initial begin
        $readmemh("verilog/weights/conv1_w.mem", conv1_w);
        $readmemh("verilog/weights/conv1_b.mem", conv1_b);
        $readmemh("verilog/weights/conv1_mul.mem", conv1_mul_mem);
        $readmemh("verilog/weights/conv2_w.mem", conv2_w);
        $readmemh("verilog/weights/conv2_b.mem", conv2_b);
        $readmemh("verilog/weights/conv2_mul.mem", conv2_mul_mem);
        $readmemh("verilog/weights/conv3_w.mem", conv3_w);
        $readmemh("verilog/weights/conv3_b.mem", conv3_b);
        $readmemh("verilog/weights/conv3_mul.mem", conv3_mul_mem);
        $readmemh("verilog/weights/fc1_w.mem", fc1_w);
        $readmemh("verilog/weights/fc1_b.mem", fc1_b);
        $readmemh("verilog/weights/fc1_mul.mem", fc1_mul_mem);
        $readmemh("verilog/weights/fc2_w.mem", fc2_w);
        $readmemh("verilog/weights/fc2_b.mem", fc2_b);
        $readmemh("verilog/weights/gap_mul.mem", gap_mul_mem);
        $readmemh("verilog/weights/geom_mul.mem", geom_mul_mem);
    end

    assign in_ready=(state==S_LOAD);
    assign busy=(state!=S_IDLE);
    reg [1:0] layer;
    reg [5:0] pool_x,pool_y,out_ch;
    reg sub_x,sub_y;
    reg [1:0] kx,ky;
    reg [5:0] in_ch;
    reg signed [31:0] conv_acc;
    reg [7:0] conv_max;
    integer in_width,in_channels,out_width,out_channels;
    integer conv_x,conv_y,conv_input_addr,conv_weight_addr;
    reg signed [7:0] conv_input_value,conv_weight_value;
    reg signed [31:0] conv_bias_value;
    reg [31:0] conv_multiplier;
    reg [7:0] conv_relu_value;

    function [7:0] requant_relu;
        input signed [31:0] value;
        input [31:0] multiplier;
        reg signed [63:0] value_ext,multiplier_ext,product_ext,shifted;
        begin
            value_ext={{32{value[31]}},value};
            multiplier_ext={32'd0,multiplier};
            product_ext=value_ext*multiplier_ext;
            shifted=product_ext >>> Q24_SHIFT;
            if (shifted<0) requant_relu=8'd0;
            else if (shifted>127) requant_relu=8'd127;
            else requant_relu=shifted[7:0];
        end
    endfunction

    function [7:0] requant_unsigned;
        input [7:0] value;
        input [31:0] multiplier;
        reg [63:0] value_ext,multiplier_ext,product_ext,shifted;
        begin
            value_ext={56'd0,value};
            multiplier_ext={32'd0,multiplier};
            product_ext=value_ext*multiplier_ext;
            shifted=product_ext >> Q24_SHIFT;
            if (shifted>127) requant_unsigned=8'd127;
            else requant_unsigned=shifted[7:0];
        end
    endfunction

    always @* begin
        in_width=64; in_channels=1; out_width=32; out_channels=8;
        case (layer)
            2'd1: begin in_width=64; in_channels=1; out_width=32; out_channels=8; end
            2'd2: begin in_width=32; in_channels=8; out_width=16; out_channels=16; end
            2'd3: begin in_width=16; in_channels=16; out_width=8; out_channels=32; end
            default: begin in_width=64; in_channels=1; out_width=32; out_channels=8; end
        endcase
        conv_x=(pool_x*2)+sub_x+kx-1;
        conv_y=(pool_y*2)+sub_y+ky-1;
        conv_input_addr=0;
        conv_weight_addr=(((out_ch*in_channels+in_ch)*3+ky)*3+kx);
        conv_input_value=8'sd0;
        conv_weight_value=8'sd0;
        conv_bias_value=32'sd0;
        conv_multiplier=32'd1;
        if ((conv_x>=0)&&(conv_x<in_width)&&(conv_y>=0)&&(conv_y<in_width)) begin
            case (layer)
                2'd1: begin
                    conv_input_addr=(conv_y*64)+conv_x;
                    conv_input_value=input_mem[conv_input_addr] ? 8'sd1 : 8'sd0;
                    conv_weight_value=conv1_w[conv_weight_addr];
                    conv_bias_value=conv1_b[out_ch];
                    conv_multiplier=conv1_mul_mem[0];
                end
                2'd2: begin
                    conv_input_addr=((conv_y*32)+conv_x)*8+in_ch;
                    conv_input_value=act1_mem[conv_input_addr];
                    conv_weight_value=conv2_w[conv_weight_addr];
                    conv_bias_value=conv2_b[out_ch];
                    conv_multiplier=conv2_mul_mem[0];
                end
                2'd3: begin
                    conv_input_addr=((conv_y*16)+conv_x)*16+in_ch;
                    conv_input_value=act2_mem[conv_input_addr];
                    conv_weight_value=conv3_w[conv_weight_addr];
                    conv_bias_value=conv3_b[out_ch];
                    conv_multiplier=conv3_mul_mem[0];
                end
                default: begin end
            endcase
        end else begin
            case (layer)
                2'd1: begin conv_weight_value=conv1_w[conv_weight_addr]; conv_bias_value=conv1_b[out_ch]; conv_multiplier=conv1_mul_mem[0]; end
                2'd2: begin conv_weight_value=conv2_w[conv_weight_addr]; conv_bias_value=conv2_b[out_ch]; conv_multiplier=conv2_mul_mem[0]; end
                2'd3: begin conv_weight_value=conv3_w[conv_weight_addr]; conv_bias_value=conv3_b[out_ch]; conv_multiplier=conv3_mul_mem[0]; end
                default: begin end
            endcase
        end
        conv_relu_value=requant_relu(conv_acc+conv_bias_value,conv_multiplier);
    end

    reg [5:0] gap_ch;
    reg [6:0] gap_pos;
    reg [15:0] gap_sum,gap_sum_rounded;
    reg [7:0] gap_average_code,gap_quantized;
    always @* begin
        gap_sum_rounded=gap_sum+{8'd0,act3_mem[gap_pos*32+gap_ch]}+16'd32;
        gap_average_code=gap_sum_rounded[13:6];
        gap_quantized=requant_unsigned(gap_average_code,gap_mul_mem[0]);
    end

    reg fc_stage;
    reg [6:0] fc_out;
    reg [5:0] fc_in;
    reg signed [31:0] fc_acc;
    reg signed [7:0] fc_input_value,fc_weight_value;
    reg signed [31:0] fc_bias_value;
    reg [31:0] fc_multiplier;
    wire signed [7:0] mac_activation,mac_weight;
    wire signed [15:0] mac_product;
    wire signed [15:0] conv_mac_product;
    assign conv_mac_product = (layer == 2'd1) ?
        (conv_input_value[0] ? {{8{conv_weight_value[7]}}, conv_weight_value} : 16'sd0) : mac_product;
    assign mac_activation=(state==S_MAC)?conv_input_value:fc_input_value;
    assign mac_weight=(state==S_MAC)?conv_weight_value:fc_weight_value;
    cnn_mac_unit u_serial_mac(.activation(mac_activation),.weight(mac_weight),.product(mac_product));

    reg [2:0] argmax_index,best_class;
    reg signed [31:0] best_logit;
    always @* begin
        fc_input_value=8'sd0;
        fc_weight_value=8'sd0;
        fc_bias_value=32'sd0;
        fc_multiplier=32'd1;
        if (fc_stage==1'b0) begin
            fc_input_value=fc_input_mem[fc_in];
            fc_weight_value=fc1_w[(fc_out*36)+fc_in];
            fc_bias_value=fc1_b[fc_out];
            fc_multiplier=fc1_mul_mem[0];
        end else begin
            fc_input_value=fc1_act_mem[fc_in];
            fc_weight_value=fc2_w[(fc_out*64)+fc_in];
            fc_bias_value=fc2_b[fc_out];
        end
    end

    integer conv_dst_index;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state<=S_IDLE; load_count<=0; geometry_latched<=0;
            layer<=1; pool_x<=0; pool_y<=0; out_ch<=0; sub_x<=0; sub_y<=0;
            kx<=0; ky<=0; in_ch<=0; conv_acc<=0; conv_max<=0;
            gap_ch<=0; gap_pos<=0; gap_sum<=0;
            fc_stage<=0; fc_out<=0; fc_in<=0; fc_acc<=0;
            argmax_index<=1; best_class<=0; best_logit<=0;
            result_valid<=0; result_class<=0;
        end else begin
            result_valid<=0;
            case (state)
                S_IDLE: if (start) begin geometry_latched<=geometry_q7; load_count<=0; state<=S_LOAD; end
                S_LOAD: if (in_valid) begin
                    input_mem[load_count]<=in_pixel;
                    if (load_count==13'd4095) begin
                        layer<=1; pool_x<=0; pool_y<=0; out_ch<=0;
                        sub_x<=0; sub_y<=0; conv_max<=0; state<=S_LAYER_SETUP;
                    end else load_count<=load_count+1'b1;
                end
                S_LAYER_SETUP: begin conv_acc<=0; kx<=0; ky<=0; in_ch<=0; state<=S_MAC; end
                S_MAC: begin
                    conv_acc<=conv_acc+{{16{conv_mac_product[15]}},conv_mac_product};
                    // PyTorch Conv2D weights are flattened OIHW: input-channel,
                    // kernel-row, then kernel-column. Traverse kx -> ky -> in_ch
                    // so the serial MAC reads weights and input channels in that order.
                    if (kx==2) begin
                        kx<=0;
                        if (ky==2) begin
                            ky<=0;
                            if (in_ch==(in_channels-1)) begin
                                in_ch<=0;
                                state<=S_CONV_FINISH;
                            end else begin
                                in_ch<=in_ch+1'b1;
                            end
                        end else begin
                            ky<=ky+1'b1;
                        end
                    end else begin
                        kx<=kx+1'b1;
                    end
                end
                S_CONV_FINISH: begin
                    if (sub_x&&sub_y) begin
                        case (layer)
                            2'd1: begin
                                conv_dst_index=((pool_y*32+pool_x)*8)+out_ch;
                                act1_mem[conv_dst_index]<=(conv_relu_value>conv_max)?conv_relu_value:conv_max;
                            end
                            2'd2: begin
                                conv_dst_index=((pool_y*16+pool_x)*16)+out_ch;
                                act2_mem[conv_dst_index]<=(conv_relu_value>conv_max)?conv_relu_value:conv_max;
                            end
                            2'd3: begin
                                conv_dst_index=((pool_y*8+pool_x)*32)+out_ch;
                                act3_mem[conv_dst_index]<=(conv_relu_value>conv_max)?conv_relu_value:conv_max;
                            end
                            default: begin end
                        endcase
                        sub_x<=0; sub_y<=0; conv_max<=0;
                        if (pool_x<(out_width-1)) begin pool_x<=pool_x+1'b1; state<=S_LAYER_SETUP; end
                        else if (pool_y<(out_width-1)) begin pool_x<=0; pool_y<=pool_y+1'b1; state<=S_LAYER_SETUP; end
                        else if (out_ch<(out_channels-1)) begin pool_x<=0; pool_y<=0; out_ch<=out_ch+1'b1; state<=S_LAYER_SETUP; end
                        else if (layer<3) begin layer<=layer+1'b1; pool_x<=0; pool_y<=0; out_ch<=0; state<=S_LAYER_SETUP; end
                        else begin gap_ch<=0; gap_pos<=0; state<=S_GAP_SETUP; end
                    end else begin
                        if (!sub_x) sub_x<=1;
                        else begin sub_x<=0; sub_y<=1; end
                        if (conv_relu_value>conv_max) conv_max<=conv_relu_value;
                        state<=S_LAYER_SETUP;
                    end
                end
                S_GAP_SETUP: begin gap_sum<=0; gap_pos<=0; state<=S_GAP_MAC; end
                S_GAP_MAC: begin
                    if (gap_pos==63) begin
                        fc_input_mem[gap_ch]<=$signed(gap_quantized);
                        if (gap_ch==31) begin
                            fc_input_mem[32]<=$signed(requant_unsigned(geometry_latched[7:0],geom_mul_mem[0]));
                            fc_input_mem[33]<=$signed(requant_unsigned(geometry_latched[15:8],geom_mul_mem[0]));
                            fc_input_mem[34]<=$signed(requant_unsigned(geometry_latched[23:16],geom_mul_mem[0]));
                            fc_input_mem[35]<=$signed(requant_unsigned(geometry_latched[31:24],geom_mul_mem[0]));
                            fc_stage<=0; fc_out<=0; state<=S_FC_SETUP;
                        end else begin gap_ch<=gap_ch+1'b1; state<=S_GAP_SETUP; end
                    end else begin
                        gap_sum<=gap_sum+{8'd0,act3_mem[gap_pos*32+gap_ch]};
                        gap_pos<=gap_pos+1'b1;
                    end
                end
                S_FC_SETUP: begin fc_acc<=0; fc_in<=0; state<=S_FC_MAC; end
                S_FC_MAC: begin
                    fc_acc<=fc_acc+{{16{mac_product[15]}},mac_product};
                    if (((fc_stage==0)&&(fc_in==35))||((fc_stage==1)&&(fc_in==63))) state<=S_FC_FINISH;
                    else fc_in<=fc_in+1'b1;
                end
                S_FC_FINISH: begin
                    if (fc_stage==0) begin
                        fc1_act_mem[fc_out]<=requant_relu(fc_acc+fc_bias_value,fc_multiplier);
                        if (fc_out==63) begin fc_stage<=1; fc_out<=0; state<=S_FC_SETUP; end
                        else begin fc_out<=fc_out+1'b1; state<=S_FC_SETUP; end
                    end else begin
                        logits_mem[fc_out]<=fc_acc+fc_bias_value;
                        if (fc_out==4) begin
                            argmax_index<=1; best_class<=0; best_logit<=logits_mem[0]; state<=S_ARGMAX;
                        end else begin fc_out<=fc_out+1'b1; state<=S_FC_SETUP; end
                    end
                end
                S_ARGMAX: begin
                    if (logits_mem[argmax_index]>best_logit) begin best_logit<=logits_mem[argmax_index]; best_class<=argmax_index; end
                    if (argmax_index==4) begin
                        if (logits_mem[argmax_index]>best_logit) result_class<=argmax_index;
                        else result_class<=best_class;
                        result_valid<=1; state<=S_IDLE;
                    end else argmax_index<=argmax_index+1'b1;
                end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule

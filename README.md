# cs3220---stencil-computation-solved
**TO GET THIS SOLUTION VISIT:** [CS3220 – Stencil Computation Solved](https://www.ankitcodinghub.com/product/cs3220-stencil-computation-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;126100&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS3220 - Stencil Computation Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
In this project, you will implement a five-point stencil FPGA accelerator kernel in vitis HLS.

Introduction

Many scientific simulations require the numerical solution of partial differential equations. Stencils are a way to approximate the derivatives. The domain is discretized into a grid and the stencil kernel is a convolution or weighted average of nearby points and evaluated for every point on the grid.

For example, if we are working with a 3D fluid simulation and we discretize each dimension into 10000 points, we have 10^12 points (one trillion) points in the grid at which to evaluate the stencil. Needless to say, this requires a lot of processing power, and executing these kernels efficiently is very important. FPGAs can be used to accelerate these computations, which we will look at in this homework assignment.

We will consider a very simple 1-dimensional first derivative 5-point stencil and consider an implementation in HLS. The equation is:

$$f'(x) approx rac{-f(x + 2h) + 8f(x + h) – 8f(x-h) + f(x-2h)}{12h}$$ where $h$ is the grid step size.

GPU Implementation

Let’s start with some representative C++ (CUDA) code for this kernel:

__global__ void cuda_stencil_5(const double *in, double *out, double h)

{

int i = blockDim.x * blockIdx.x + threadIdx.x + 2;

out[i] = (-in[i+2] + 8*in[i+1] – 8*in[i-1] + in[i-2]) / (12 * h);

}

We will ignore the boundary for now and assume that we only execute on points with 2 left and right neighbors. The easiest way to handle this is to allocate the input array with 4 extra points and fill them with padding zeros. We do not want to add branches (if statements) into the code.

CUDA provides a grid abstraction such that you can imagine this code running for each point in the grid in a loop. The blockIdx and threadIdx provide the grid coordinates for the kernel. Translating this back to sequential code involves wrapping it in a loop. For example:

void stencil5(const double *in, double *out, double h, int size)

{

for (int i=2; i &lt; size-2; i++)

out[i] = (-in[i+2] + 8*in[i+1] – 8*in[i-1] + in[i-2]) / (12 * h);

}

The sample cpp code for five-point stencil has been provided in stencil.cpp in the c_stencil subfolder. In the provided example, we sample 128 points from linear function y=2x by the step of h=0.05. The results are the derivative of the function y=2x, which is 2.

Deliverable 1: The first part of this assignment is (1) understand how c_stencil/stencil.cpp works (2) modify hls_template/hls_stencil_shift.cpp to write a HLS version of stencil kernel (3) your HLS code should pass C_simulation, C_synthesis and C/RTL Co-sim. (4) You need to benchmark the performance of your HLS design following the same instructions that you learn in HOMEWORK #5.

Note that: We provide the tcl file to run the flow in the terminal without GUI in ./hls_template/test_hls.tcl. Here are steps to run it. cd

&lt;path_to_GT-CS3220.GITHUB.IO&gt;/Spring_2023/project3/hls_template vitis_hls -f test_hls.tcl Then the report will show up in the path ./hls_template/project3/solution1/sim/report/stencil5_cosim.rpt.

FPGA Optimization

The code above performs very well on the SIMT architecture of the GPU. The computation is “embarassingly parallel,” the GPU grid abstraction maps very naturally and directly to the underlying simulation grid, and the caches exploit data reuse effectively.

FPGA implementation, however, is a little lower level, and the same code simply running through the HLS tool works correctly but does not perform nearly as well as the FPGA is capable of. Instead, we need to use a different design, using shift registers.

Shift Registers

A shift register is an array of registers which each hold a piece of data (bits, bytes, words, etc..) and each register is connected to the next in a chain. In a single cycle/operation the data in each register is “shifted” into the next register in the chain. In a traditional CPU, this sort of bulk shift operation would have to be executed by a series of mov instructions and would take many cycles (ignoring any special instructions designed specifically for this use case), but in an FPGA all the data movement is easily done in parallel in a single cycle.

With this building block, then, we can consider a different way to implement a stencil kernel. Instead of the data-parallel model where we run the same code on different data, we can think of a the data as flowing in a stream through the shift register. Each cycle we can perform one stencil computation, shift, and push one new input element into the shift register.

Deliverable 2: (1) optimize your HLS code with shift register to improve the performance. (2) compare the performance between optimized implementation and the previous section. The performance should be measured through the C/RTL co-simulation latency (referring to hw5.md for detail). For implementation of shift register, you could refer to the Xilinx documentation for help Xilinx Shift Register.

Questions

1. Why is the shift register version more efficient than the original code for the FPGA? Why is the GPU/CPU able to execute the original version efficiently anyway? [hint] Thinking of how CPU and GPU improve the throughput for parallel code.

2. Can you think whether the HLS Pragma UNROLL could benefit the five-stage stencil design and explain the reason?

Submitting

1. [code] two versions of hls_stencil_shift.cpp; One version needs to pass the C_simulation, C_synthesis and C/RTL Co-sim; The second version needs to use shift-register to implement the stencil kernel in HLS.

2. [report] Measure the performance of both and compare the two approaches in a short write-up.

3. [Answers to questions] Try to address the questions above.

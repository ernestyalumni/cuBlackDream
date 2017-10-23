/**
 * @file   : testAxon_act.cu
 * @brief  : test Axon_act derived class with CUDA C++14, CUBLAS, CUDA Unified Memory Management
 * @details :   
 * 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171023  
 * @ref    : 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * nvcc -std=c++14 -lcublas ../src/Axon/Axon.cu RModule.cu -o Rmodule.exe
 * nvcc -arch='sm_52' -std=c++14 -lcublas ../src/Axon/Axon.o ../src/Axon/activationf.o testAxon_act.cu -o testAxon_act.exe
 * */
#include "../src/Axon/Axon.h"				// Axon_act

#include <iostream>

int main(int argc, char* argv[]) { 


}

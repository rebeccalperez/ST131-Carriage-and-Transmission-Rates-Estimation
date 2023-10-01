data {
       int<lower=1> K;  					
       
       int<lower=0> T1;  					// T: number of samples during time-intervals
       int<lower=0> T2;  					
       int<lower=0> T3;
       int<lower=0> T4;
       int<lower=0> T5;
       int<lower=0> T6;

  
       int<lower=0> N1;						//N: Number of subjects with observations in first interval
       int<lower=0> N2;	
       int<lower=0> N3;
       int<lower=0> N4;
       int<lower=0> N5;
       int<lower=0> N6;
  
    
       int<lower=1,upper=K>	y1[N1,T1]; 		// observation sequences
       int<lower=1,upper=K>	y2[N2,T2]; 		// observation sequences
       int<lower=1,upper=K>	y3[N3,T3];
       int<lower=1,upper=K>	y4[N4,T4];
       int<lower=1,upper=K>	y5[N5,T5];
       int<lower=1,upper=K>	y6[N6,T6];

      
       int<lower=0> delta_T1[N1];			//vector of time intervals between time points 1&2
       int<lower=0> delta_T2[N2];			//vector of time intervals between time points 2& 3
       int<lower=0> delta_T3[N3];			
       int<lower=0> delta_T4[N4];
       int<lower=0> delta_T5[N5];
       int<lower=0> delta_T6[N6];


       int<lower=0>	D;						//Number of covariates
       vector[D]	X1[N1]; 				//vector of covariates at time point 2
       vector[D]	X2[N2]; 				// vector of covariates at time point 3
       vector[D]	X3[N3];
       vector[D]	X4[N4];
       vector[D]	X5[N5];
       vector[D]	X6[N6];
  
    }

  
parameters {
	
   	 vector[D] beta12; //vector of predictor coefficients for state1 to state 2 transition prob
   	 vector[D] beta21;	//vector of predictor coefficients for state2 to state1 transition prob
 }
 
 transformed parameters{
 	
 	matrix[K,K]	theta_m1[N1];			//probability matrix for interval1 
 	matrix[K,K]	theta_m2[N2];			//probability matrix for interval2
 	matrix[K,K]	theta_m3[N3];
 	matrix[K,K]	theta_m4[N4];
 	matrix[K,K]	theta_m5[N5];
 	matrix[K,K]	theta_m6[N6];

 	matrix[K,K] delta_theta1[N1]; 		//vector of per sample1 and sample2 time-interval transition matrices
 	matrix[K,K] delta_theta2[N2];		//vector of per sample1 and sample2 time-interval transition matrices
 	matrix[K,K] delta_theta3[N3];
 	matrix[K,K] delta_theta4[N4];
 	matrix[K,K] delta_theta5[N5];
 	matrix[K,K] delta_theta6[N6];


 	matrix[K,K]	delta_theta_norm1[N1];	//normalised delta_theta1 so that row sum is 1
 	matrix[K,K]	delta_theta_norm2[N2];	//normalised delta_theta2 so that row sum is 1
 	matrix[K,K]	delta_theta_norm3[N3];
 	matrix[K,K]	delta_theta_norm4[N4];
 	matrix[K,K]	delta_theta_norm5[N5];
 	matrix[K,K]	delta_theta_norm6[N6];
 		
 		
 	for(i in 1:N1){
 			theta_m1[i,1,2]=exp(X1[i,1]*beta12[1]+X1[i,2]*beta12[2]);
  			theta_m1[i,1,1]=-theta_m1[i,1,2];
  			theta_m1[i,2,1]=exp(X1[i,1]*beta21[1]+X1[i,2]*beta21[1]);
  			theta_m1[i,2,2]=-theta_m1[i,2,1];
  			
  			delta_theta1[i]=matrix_exp(delta_T1[i]*theta_m1[i,,]); //Compute the TP1 & TP2 interval probabilities
  		}	
  		
  	for(i in 1:N1)
 		for(j in 1:K)
 			for(l in 1:K)
 			delta_theta_norm1[i,j,l]=delta_theta1[i,j,l]/sum(delta_theta1[i,j,]); //normalising step so that row sum for each matrix is 1.
 			
 			
 	for(i in 1:N2){
 			theta_m2[i,1,2]=exp(X2[i,1]*beta12[1]+X2[i,2]*beta12[2]);
  			theta_m2[i,1,1]=-theta_m2[i,1,2];
  			theta_m2[i,2,1]=exp(X2[i,1]*beta21[1]+X2[i,2]*beta21[2]);
  			theta_m2[i,2,2]=-theta_m2[i,2,1];
  			
  			delta_theta2[i]=matrix_exp(delta_T2[i]*theta_m2[i,,]); //Compute the TP1 & TP2 interval probabilities
  		}	
 	
 	for(i in 1:N2)
 		for(j in 1:K)
 			for(l in 1:K)
 			delta_theta_norm2[i,j,l]=delta_theta2[i,j,l]/sum(delta_theta2[i,j,]); //normalising step so that row sum for each indiv. prob matrix is 1.


for(i in 1:N3){
 			theta_m3[i,1,2]=exp(X3[i,1]*beta12[1]+X3[i,2]*beta12[2]);
  			theta_m3[i,1,1]=-theta_m3[i,1,2];
  			theta_m3[i,2,1]=exp(X3[i,1]*beta21[1]+X3[i,2]*beta21[2]);
  			theta_m3[i,2,2]=-theta_m3[i,2,1];
  			
  			delta_theta3[i]=matrix_exp(delta_T3[i]*theta_m3[i,,]); //Compute the TP1 & TP2 interval probabilities
  		}	
 	
 	for(i in 1:N3)
 		for(j in 1:K)
 			for(l in 1:K)
 			delta_theta_norm3[i,j,l]=delta_theta3[i,j,l]/sum(delta_theta3[i,j,]); //normalising step so that row sum for each indiv. prob matrix is 1.
			

for(i in 1:N4){
 			theta_m4[i,1,2]=exp(X4[i,1]*beta12[1]+X4[i,2]*beta12[2]);
  			theta_m4[i,1,1]=-theta_m4[i,1,2];
  			theta_m4[i,2,1]=exp(X4[i,1]*beta21[1]+X4[i,2]*beta21[2]);
  			theta_m4[i,2,2]=-theta_m4[i,2,1];
  			
  			delta_theta4[i]=matrix_exp(delta_T4[i]*theta_m4[i,,]); //Compute the TP1 & TP2 interval probabilities
  		}	

 	for(i in 1:N4)
 		for(j in 1:K)
 			for(l in 1:K)
 			delta_theta_norm4[i,j,l]=delta_theta4[i,j,l]/sum(delta_theta4[i,j,]); //normalising step so that row sum for each indiv. prob matrix is 1.

for(i in 1:N5){
 			theta_m5[i,1,2]=exp(X5[i,1]*beta12[1]+X5[i,2]*beta12[2]);
  			theta_m5[i,1,1]=-theta_m5[i,1,2];
  			theta_m5[i,2,1]=exp(X5[i,1]*beta21[1]+X5[i,2]*beta12[2]);
  			theta_m5[i,2,2]=-theta_m5[i,2,1];
  			
  			delta_theta5[i]=matrix_exp(delta_T5[i]*theta_m5[i,,]); //Compute the TP1 & TP2 interval probabilities
  		}	
 	
 	for(i in 1:N5)
 		for(j in 1:K)
 			for(l in 1:K)
 			delta_theta_norm5[i,j,l]=delta_theta5[i,j,l]/sum(delta_theta5[i,j,]); //normalising step so that row sum for each indiv. prob matrix is 1.
			

for(i in 1:N6){
 			theta_m6[i,1,2]=exp(X6[i,1]*beta12[1]+X6[i,2]*beta12[2]);
  			theta_m6[i,1,1]=-theta_m6[i,1,2];
  			theta_m6[i,2,1]=exp(X6[i,1]*beta21[1]+X6[i,2]*beta21[2]);
  			theta_m6[i,2,2]=-theta_m6[i,2,1];
  			
  			delta_theta6[i]=matrix_exp(delta_T6[i]*theta_m6[i,,]); //Compute the TP1 & TP2 interval probabilities
  		}	
 	
 	for(i in 1:N6)
 		for(j in 1:K)
 			for(l in 1:K)
 			delta_theta_norm6[i,j,l]=delta_theta6[i,j,l]/sum(delta_theta6[i,j,]); //normalising step so that row sum for each indiv. prob matrix is 1.
}

      
    model {
   
    	for(d in 1:D){
    		beta12[d]~normal(-4,2);
    		beta21[d]~normal(-4,2);
    		}

	
	for(i in 1:N1)
		for(t in 2:T1)
        	y1[i,t] ~ categorical(to_vector(delta_theta_norm1[i,y1[i,t-1]]));
        	
    for(i in 1:N2)
    	for(t in 2:T2)   		
        	y2[i,t] ~ categorical(to_vector(delta_theta_norm2[i,y2[i,t-1]])); 
        	
    for(i in 1:N3)
    	for(t in 2:T3)   		
        	y3[i,t] ~ categorical(to_vector(delta_theta_norm3[i,y3[i,t-1]]));
        	
    for(i in 1:N4)
    	for(t in 2:T4)   		
        	y4[i,t] ~ categorical(to_vector(delta_theta_norm4[i,y4[i,t-1]]));
        	
    for(i in 1:N5)
    	for(t in 2:T5)   		
        	y5[i,t] ~ categorical(to_vector(delta_theta_norm5[i,y5[i,t-1]]));
        	
    for(i in 1:N6)
    	for(t in 2:T6)   		
        	y6[i,t] ~ categorical(to_vector(delta_theta_norm6[i,y6[i,t-1]]));
    
     }

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define FALSE 0
#define TRUE 1

FILE * fpOut;

time_t start_time;

#define MAX_LINE_LEN 512 //TODO - Make configurable at runtime
char *line; // The max input file line length is arbitrarily set to 512 

long random_seed = 0L; //Set in main() below.

int NUM_TRAINING_REPETITIONS = 3000; //TODO - Make configurable at runtime

#define numInputUnits 40 	//TODO - Make configurable at runtime
#define numHiddenUnits 10 	//TODO - Make configurable at runtime
#define numOutputUnits 1	//TODO - Make configurable at runtime
const float c = 0.25;  // learning rate  //TODO - Make configurable at runtime

#define MAX_NUM_EXAMPLES 10000	//TODO - Make configurable at runtime

double inputs[numInputUnits];
double weightsLayerOne[numInputUnits][numHiddenUnits];
double weightsLayerTwo[numHiddenUnits][numOutputUnits];
double weightsHiddenUnitsBias[numHiddenUnits];
double weightsOutputUnitsBias[numOutputUnits];
double hiddenLayerOutput[numHiddenUnits];
double outputLayerOutput[numOutputUnits];

int showConcept = FALSE;

int enteringEXAMPLES = FALSE;
int enteringPOSITIVES = FALSE;
char trainExamplesPOS[MAX_NUM_EXAMPLES][MAX_LINE_LEN]; //TODO - Make configurable at runtime
char trainExamplesNEG[MAX_NUM_EXAMPLES][MAX_LINE_LEN]; //TODO - Make configurable at runtime
char testExamplesPOS[MAX_NUM_EXAMPLES][MAX_LINE_LEN];  //TODO - Make configurable at runtime
char testExamplesNEG[MAX_NUM_EXAMPLES][MAX_LINE_LEN];  //TODO - Make configurable at runtime
int numTrainPositives = 0;
int numTrainNegatives = 0;
int numTestPositives = 0;
int numTestNegatives = 0;

void loadInputs(char *s);

double rand0to1(void)
{
	// The standard rand() function returns a value between 0 and RAND_MAX.
	// We need a value between 0-1, so this just divides by the range
	// to get a value we can use.
	
	double rnd = rand();
	return (rnd / RAND_MAX);
}

void reset(void)
{
	srand(random_seed);
	//double r = rand0to1();
	
	#pragma acc loop
	for(int j=0; j < numHiddenUnits; j++) 
	{
	    #pragma acc loop
	    for(int i=0; i< numInputUnits; i++)
	    {
	       weightsLayerOne[i][j] = rand0to1();
	    }
	    #pragma acc loop
	    for(int k=0; k<numOutputUnits; k++)
	    {
	       weightsLayerTwo[j][k] = rand0to1();
	    }
    }
	#pragma acc loop
	for(int j=0; j < numHiddenUnits; j++) 
	{
		weightsHiddenUnitsBias[j] =  rand0to1(); 
	}
	
	#pragma acc loop
	for(int k=0; k < numOutputUnits; k++) 
	{
		weightsOutputUnitsBias[k] =  rand0to1(); 
	}
}


int LTU(double input)
{ 
	if	(input > 0.5) 
		return 1;  
	else 
		return 0;  
}

double sigmoid(double input)
{  
	return 1.0/(1.0 + pow(M_E, -input));
}

void runNet(void)
{
    double summedInput;
	
    #pragma acc loop
    for(int j=0; j < numHiddenUnits; j++) 
    {
		summedInput = 0.0;
		for(int i=0; i<numInputUnits; i++)
		{
			summedInput += weightsLayerOne[i][j] * inputs[i];
		}
		summedInput +=  weightsHiddenUnitsBias[j];
		hiddenLayerOutput[j] = sigmoid(summedInput);
	}
	for(int k=0; k < numOutputUnits; k++) 
	{
		summedInput = 0.0;
		for(int j=0; j<numHiddenUnits; j++)
		{
		     summedInput += weightsLayerTwo[j][k] * hiddenLayerOutput[j];
		}
		summedInput +=  weightsOutputUnitsBias[k];
		outputLayerOutput[k] = sigmoid(summedInput);
	}	
}

void trainOneOutputUnitOnOneExampleForOneEpoch(int k, double d)
{

	runNet();
	double f = outputLayerOutput[k];
	double deltaK = (d-f)*f*(1-f);
	double deltaJ;

    //  First update the weight on the connection from output unit k's bias
	weightsOutputUnitsBias[k] += c*deltaK;
	// Then update the weights going into output unit k from hidden units.
	for(int j=0; j<numHiddenUnits; j++)  
	{
	     // First backprop the error from output unit k to hidden layer unit j.
	         //  First calculate the backpropped error deltaJ.
	         deltaJ = hiddenLayerOutput[j]*(1-hiddenLayerOutput[j])*(deltaK*weightsLayerTwo[j][k]);
	         // Then update the weight of the bias going into hidden unit j.
	     	weightsHiddenUnitsBias[j] += c*deltaJ;
	         //  Then update the weights on the connections from the input units
	         //  into hidden unit j.
	     	 for(int i=0; i < numInputUnits; i++)
	     	 {
	     	    weightsLayerOne[i][j] += c*deltaJ*inputs[i];
	     	 };
	     // Lastly go back and update the weight going from hidden unit j to output unit k.
	     weightsLayerTwo[j][k] +=  c*deltaK*hiddenLayerOutput[j];
	}
}

/*  TODO - Delete!
void trainOneOutputUnitOnOneExampleForMultipleEpochs(int k, double label, int numEpochs)
{
	for(int i=0; i< numEpochs; i++)
	{
		trainOneOutputUnitOnOneExampleForOneEpoch(k,label);
	}
}
*/
 
void readTrainingSet(void)  
{
	FILE *fp;
	
	numTrainPositives = 0;
	
	fp = fopen("trainPOS.txt", "r"); //TODO - allow for configuration at runtime
		
	if (fp != NULL )
	{
		while (fgets(line, 256, fp) != NULL)  //<<---- TODO - Start here, program blows up
		{  //  Get input vectors from file one line at a time until end of file
			//TODO - convert to binary immediately, don't store strings
			strcpy(&trainExamplesPOS[numTrainPositives++][0], line);
			//printf("+");
		}
		fclose(fp);
	}
	else
	{
		printf("FAILED TO OPEN trainPOS.txt\n");
	}

	numTrainNegatives = 0;
	fp = fopen("trainNEG.txt", "r"); //TODO - allow for configuration at runtime
	if (fp != NULL )
	{
		while (fgets(line, 256, fp) != NULL)  
		{  //  Get input vectors from file one line at a time until end of file
			//TODO - convert to binary immediately, don't store strings
			strcpy(&trainExamplesNEG[numTrainNegatives++][0], line);
			//printf("-");
		}
		fclose(fp);
	}
	else
	{
		printf("FAILED TO OPEN trainNEG.txt\n");
	}
	printf("\n");
}

void readTestingSet(void)  
{
	FILE *fp;
	
	// Read test positives
	numTestPositives = 0;
	fp = fopen("testPOS.txt", "r"); //TODO - allow for configuration at runtime
	if (fp != NULL )
	{
		while (fgets(line, 256, fp) != NULL)  
		{  //  Get input vectors from file one line at a time until end of file
			//TODO - convert to binary immediately, don't store strings
			strcpy(&testExamplesPOS[numTestPositives++][0], line);
			//printf("x");
		}
		fclose(fp);
	}
	else
	{
		printf("FAILED TO OPEN testPOS.txt\n");
	}

	// Read test negatives
	numTestNegatives = 0;
	fp = fopen("testNEG.txt", "r"); //TODO - allow for configuration at runtime
	if (fp != NULL )
	{
		while (fgets(line, 256, fp) != NULL)  
		{  //  Get input vectors from file one line at a time until end of file
			//TODO - convert to binary immediately, don't store strings
			strcpy(&testExamplesNEG[numTestNegatives++][0], line);
			//printf("/");
		}
		fclose(fp);
	}
	else
	{
		printf("FAILED TO OPEN testNEG.txt\n");
	}
	printf("\n");
}
	
void train(void)  
{
    int num = 0;
    int p = 0;
    //int c = 0;
    //int dim = 0;
    //char *s;

	//printf("BEGIN TRAINING\n");
	for(num=0; num<NUM_TRAINING_REPETITIONS; num++)  
	{
		for(p=0;p<numTrainPositives; p++)  
		{			
			loadInputs(trainExamplesPOS[p]);

			trainOneOutputUnitOnOneExampleForOneEpoch(0,1);
			
			loadInputs(trainExamplesNEG[p]);
			
			trainOneOutputUnitOnOneExampleForOneEpoch(0,0);
    	}
	  }
}
	  
int charToBit(char ch) 
{
	if(ch == '0') 
		return 0; 
	else return 1; 
}

// TODO - Do this as the data is loaded rather than as a separate step.
void loadInputs(char *s)  
{

	int dim = strlen(s) - 1;
	for(int c = 0; c < dim; c++)  
	{
		inputs[c]= charToBit(s[c]); 
	}
}

void test(void)  
{
	int p;
	int testPOSscore, testNEGscore,trainPOSscore, trainNEGscore;
	int testPOSscorePCT, testNEGscorePCT, trainPOSscorePCT, trainNEGscorePCT;	int POSscore, NEGscore, testScore, trainScore;
	int POSscorePCT, NEGscorePCT, testScorePCT, trainScorePCT;
	int allExamplesScore, allExamplesScorePCT;
 	
	testPOSscore=0;
	for(p=0;p<numTestPositives; p++)  
	{
		 loadInputs(testExamplesPOS[p]);  
		 runNet();
		 if(LTU(outputLayerOutput[0]) == 1) 
		 	testPOSscore++;
	}
	testPOSscorePCT = (int) ((100.0*testPOSscore)/numTestPositives);
	
	testNEGscore=0;
	for(p=0;p<numTestNegatives; p++)  
	{
		 loadInputs(testExamplesNEG[p]);  
		 runNet();
		 if(LTU(outputLayerOutput[0]) == 0) 
		 	testNEGscore++; 
	}
	testNEGscorePCT = (int) ((100.0*testNEGscore)/numTestNegatives);
	
 	trainPOSscore=0;
 	for(p=0;p<numTrainPositives; p++)  
 	{
		loadInputs(trainExamplesPOS[p]);  
		runNet();
	 	if(LTU(outputLayerOutput[0]) == 1) 
	 		trainPOSscore++;
	}
 	trainPOSscorePCT = (int) ((100.0*trainPOSscore)/numTrainPositives);

 	trainNEGscore=0;
 	for(p=0;p<numTrainNegatives; p++)  
 	{
		loadInputs(trainExamplesNEG[p]);  
		runNet();
		if(LTU(outputLayerOutput[0]) == 0) 
			trainNEGscore++; 
	}

 	trainNEGscorePCT = (int) ((100.0*trainNEGscore)/numTrainNegatives);
 	trainScore = trainPOSscore+trainNEGscore;
 	testScore = testPOSscore+testNEGscore;
 	POSscore = trainPOSscore+testPOSscore;
 	NEGscore = trainNEGscore+testNEGscore;
 	trainScorePCT = (int) ((100.0*trainScore)/(numTrainPositives+numTrainNegatives));
 	testScorePCT = (int) ((100.0*testScore)/(numTestPositives+numTestNegatives));
 	POSscorePCT = (int) ((100.0*POSscore)/(numTrainPositives+numTestPositives));
 	NEGscorePCT = (int) ((100.0*NEGscore)/(numTrainNegatives+numTestNegatives));
 	allExamplesScore = trainScore+testScore;
 	allExamplesScorePCT = (int) ((100.0 * allExamplesScore)/(numTrainPositives+numTrainNegatives+numTestPositives+numTestNegatives));
	printf("\n\n");
 	printf("                                RANDOM SEED:   %ld\n\n", random_seed);
 	printf("                      NUMBER OF INPUT UNITS:   %d\n", numInputUnits);
 	printf("                     NUMBER OF HIDDEN UNITS:   %d\n", numHiddenUnits);
 	printf("                     NUMBER OF OUTPUT UNITS:   %d\n", numOutputUnits);
 	printf("                        TRAINING ITERATIONS:   %d\n", NUM_TRAINING_REPETITIONS);
 	printf("\n");
 	printf("                      |   POSITIVES    |    NEGATIVES    |   ALL EXAMPLES   |\n");
 	printf("                      |-----------------------------------------------------|\n");
 	printf("    TRAINING SET      |      %3d%%      |       %3d%%      |      %3d%%        |\n", trainPOSscorePCT, trainNEGscorePCT, trainScorePCT);
 	printf("    TEST     SET      |      %3d%%      |       %3d%%      |      %3d%%        |\n", testPOSscorePCT, testNEGscorePCT, testScorePCT);
 	printf(" ---------------------|----------------|-----------------|------------------|\n");
 	printf("    OVER ALL EXAMPLES |      %3d%%      |       %3d%%      |      %3d%%        |\n", POSscorePCT, NEGscorePCT, allExamplesScorePCT);
 	printf("\n\n");

 	fprintf(fpOut, "%ld, %d, %d, %d, %d, %3d, %3d, %3d, %3d, %3d, %3d, %3d, %3d, %3d\n",
 			random_seed,
			numInputUnits,
			numHiddenUnits,
			numOutputUnits,
			NUM_TRAINING_REPETITIONS,
			trainPOSscorePCT,
			trainNEGscorePCT,
			trainScorePCT,
			testPOSscorePCT,
			testNEGscorePCT,
			testScorePCT,
			POSscorePCT,
			NEGscorePCT,
			allExamplesScorePCT);
}


/*
void results(void)  
{
	printf("Num positive training examples:  %d", numTrainPositives);
	loadInputs(trainExamplesPOS[0]);
	printf("%lf, %lf, %lf, %lf, %lf", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
}
*/

int main(int argc, char ** argv)
{
	char time_buffer[32];
	start_time = time(NULL);
	
	printf("\n\n");
	printf("                                   Backprop.c\n");

	if ((line = (char *)malloc(sizeof(char) * MAX_LINE_LEN)) == NULL)
	{
		printf("malloc failed\n");
		return 100;
	}
	
	readTrainingSet();	
	readTestingSet();
		
	// Output file
	fpOut = fopen("Results.txt", "w");
	if (fpOut == NULL )
	{
		printf("ERROR OPENING Results.txt\n");
	}

	//TODO - Make this configurable at runtime       
	for(random_seed = 1; random_seed <= 10000; random_seed+=1)   //102,112,122,132,142,152,162,172,182,192,
	{
		reset();
		train();
		test();
	} // end of for loop
	
	fclose(fpOut);

	free(line);
	
	sprintf(time_buffer, "%6.0f", difftime(time(NULL), start_time));
	printf("Runtime: %s seconds\n", time_buffer); 

	return 0;
}

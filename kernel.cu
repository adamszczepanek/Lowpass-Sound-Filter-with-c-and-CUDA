#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <time.h>
#include <ctime>

using namespace std;

typedef struct  WAV_HEADER
{
	/* RIFF Chunk Descriptor */
	uint8_t         RIFF[4];        // RIFF Header Magic header
	uint32_t        ChunkSize;      // RIFF Chunk Size
	uint8_t         WAVE[4];        // WAVE Header
	/* "fmt" sub-chunk */
	uint8_t         fmt[4];         // FMT header
	uint32_t        Subchunk1Size;  // Size of the fmt chunk
	uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
	uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Sterio
	uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
	uint32_t        bytesPerSec;    // bytes per second
	uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
	uint16_t        bitsPerSample;  // Number of bits per sample
	/* "data" sub-chunk */
	uint8_t         Subchunk2ID[4]; // "data"  string
	uint32_t        Subchunk2Size;  // Sampled data length
} wav_hdr;

// Function prototypes
int getFileSize(FILE* inFile);
double* filter(wav_hdr wavHeader, double limit_freq);

__global__ void filterr(int8_t* buffer_d, int8_t* buffer_dd, double* filtr)				//a gpu function that calculates the convolution
{																						//for each sample represented by each thread
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double suma = 0;
	if (i > 30)
	{
		for (int j = 0; j < 31; j++)
			suma += filtr[j] * abs(buffer_d[i - j]);
	}
	else 
		for (int k = 0; k < i; k++)
			suma += filtr[k] * abs(buffer_d[i - k]);
	
	buffer_dd[i] = suma;
}

int main(int argc, char* argv[])
{
	wav_hdr wavHeader;
	wav_hdr* wavHeader_d;
	wav_hdr* wavHeaderPtr = &wavHeader;
	wav_hdr* wavHeaderPtr1 = &wavHeader;
	int headerSize = sizeof(wav_hdr);
	int filelength = 0;
	const char* filePath;
	string input;
	if (argc <= 1)
	{
		cout << "Input wave file name: ";
		cin >> input;
		cin.get();
		filePath = input.c_str();
	}
	else
	{
		filePath = argv[1];
		cout << "Input wave file name: " << filePath << endl;
	}

	FILE* wavFile = fopen(filePath, "r");
	FILE* wavFile_d = fopen(filePath, "r");
	FILE* output = fopen("output.wav", "w");
	if (wavFile == nullptr)
	{
		fprintf(stderr, "Unable to open wave file: %s\n", filePath);
		return 1;
	}

	//Read the header
	size_t bytesRead = fread(&wavHeader, 1, headerSize, wavFile);
	size_t bytesWritten = fwrite(wavHeaderPtr1, sizeof(wav_hdr), 1, output);

	static const uint64_t BUFFER_SIZE = wavHeader.Subchunk2Size;
	/*cudaMalloc((void**)&wavHeader_d, sizeof(wav_hdr));
	cudaMemcpy(wavHeader_d, wavHeaderPtr, BUFFER_SIZE * sizeof(int8_t), cudaMemcpyHostToDevice);*/

	cout << "Header Read " << bytesRead << " bytes." << endl;

	if (bytesRead > 0)
	{
		//Read the data
		int8_t* buffer = new int8_t[BUFFER_SIZE];
		int8_t* buffer_d = new int8_t[BUFFER_SIZE];
		int8_t* buffer_dd = new int8_t[BUFFER_SIZE];
		double* filtr_cpu = new double[31];
		double* filtr_gpu = new double[31];
		
		filtr_cpu = filter(wavHeader, 10000);

		cudaMalloc((void**)&buffer_d, BUFFER_SIZE * sizeof(int8_t));
		cudaMalloc((void**)&buffer_dd, BUFFER_SIZE * sizeof(int8_t));
		cudaMalloc((void**)&filtr_gpu, 31 * sizeof(double));

		while ((bytesRead = fread(buffer, sizeof buffer[0], BUFFER_SIZE, wavFile)) > 0)
		{
			cout << "data bytes read: " << bytesRead << endl;
		}
		//allocate memory on GPU
		cudaMemcpy(buffer_d, buffer, BUFFER_SIZE * sizeof(int8_t), cudaMemcpyHostToDevice);
		cudaMemcpy(filtr_gpu, filtr_cpu, 31 * sizeof(double), cudaMemcpyHostToDevice);

		const int size_blocks = 1024;
		int num_blocks = BUFFER_SIZE / size_blocks - 1;
		//calling out GPU function(kernel)
		filterr <<<num_blocks, size_blocks >>> (buffer_d, buffer_dd, filtr_gpu);
		//transfering data from device to host
		cudaMemcpy(buffer, buffer_dd, BUFFER_SIZE * sizeof(int8_t), cudaMemcpyDeviceToHost);
		//writing calculated data to a file
		fwrite(buffer, wavHeader.Subchunk2Size, 1, output);

		cudaFree(buffer_d);
		cudaFree(buffer_dd);
		cudaFree(filtr_gpu);
		
		
		delete[] buffer;
		buffer = nullptr;
		filelength = getFileSize(wavFile);

		cout << "File is                    :" << filelength << " bytes." << endl;
		cout << "RIFF header                :" << wavHeader.RIFF[0] << wavHeader.RIFF[1] << wavHeader.RIFF[2] << wavHeader.RIFF[3] << endl;
		cout << "WAVE header                :" << wavHeader.WAVE[0] << wavHeader.WAVE[1] << wavHeader.WAVE[2] << wavHeader.WAVE[3] << endl;
		cout << "FMT                        :" << wavHeader.fmt[0] << wavHeader.fmt[1] << wavHeader.fmt[2] << wavHeader.fmt[3] << endl;
		cout << "Data size                  :" << wavHeader.ChunkSize << endl;


		cout << "Sampling Rate              :" << wavHeader.SamplesPerSec << endl;
		cout << "Number of bits used        :" << wavHeader.bitsPerSample << endl;
		cout << "Number of channels         :" << wavHeader.NumOfChan << endl;
		cout << "Number of bytes per second :" << wavHeader.bytesPerSec << endl;
		cout << "Data length                :" << wavHeader.Subchunk2Size << endl;
		cout << "Audio Format               :" << wavHeader.AudioFormat << endl;


		cout << "Block align                :" << wavHeader.blockAlign << endl;
		cout << "Data string                :" << wavHeader.Subchunk2ID[0] << wavHeader.Subchunk2ID[1] << wavHeader.Subchunk2ID[2] << wavHeader.Subchunk2ID[3] << endl;

	}
	fclose(wavFile);
	fclose(output);

	return 0;
}

// find the file size
int getFileSize(FILE* inFile)
{
	int fileSize = 0;
	fseek(inFile, 0, SEEK_END);

	fileSize = ftell(inFile);

	fseek(inFile, 0, SEEK_SET);
	return fileSize;
}
//calculating lowpass filter impulse repsonse
double* filter(wav_hdr wavHeader, double limit_freq)
{
	double sampling_freq = wavHeader.SamplesPerSec;
	double usr_freq = limit_freq / sampling_freq / 2;
	int il_probek = 31;
	double* filtr = new double[il_probek];
	int n = 0;

	for(int i = -il_probek/2; i < 0; i++)
	{
		filtr[n] = sin(2 * 3.1415 * usr_freq * i) / (3.1415 * i);
		n++;
	}
	filtr[n] = 2 * usr_freq;
	n++;
	for (int j = 1; j <= il_probek / 2; j++)
	{
		filtr[n] = sin(2 * 3.1415 * usr_freq * j) / (3.1415 * j);
		n++;
	}
	return filtr;
}
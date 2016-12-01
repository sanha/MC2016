#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "timers.h"
#include "mpi.h"

int print_matrix = 0;
int validation = 0;

void mat_mul( float *c, float *a, float *b, int NDIM, int rows) {
	int i, j, k;
	
	for (i = 0; i < rows; i++) {
		for (j = 0; j < NDIM; j++) {
			for (k = 0; k < NDIM; k++) {
				c[i * NDIM + k] += a[i * NDIM + j] * b[j * NDIM + k];
			}
		}
	}

	return ;
}

void check_mat_mul( float * c, float * a, float * b, int NDIM )
{
	int i, j, k, x, y;
	float sum, result;
	int validated = 1;

	printf("Validating the result..\n");
	
  srand(time(NULL));
	// C = AB
	for( x = 0; x < 10; x++ )
	{
		for( y = 0; y < 10; y++ )
		{
      i = rand() % NDIM;
      j = rand() % NDIM;
			sum = 0;

			for( k = 0; k < NDIM; k++ )
			{
				sum += a[i * NDIM + k] * b[k * NDIM + j];
			}

      result = c[i * NDIM + j];

			if( result - sum > 0.001 || result - sum < -0.001 )
			{
				printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i * NDIM + j], sum );
				validated = 0;
			}
		}
	}

	printf("Validation : ");
	if( validated )
		printf("SUCCESSFUL.\n");
	else
		printf("FAILED.\n");
}

void print_mat( float * mat, int NDIM )
{
	int i, j;

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			printf("%8.2lf ", mat[i * NDIM + j]);
		}
		printf("\n");
	}
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [-pvh]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -p : print matrix data.\n");
	printf("  -v : validate matrix multiplication.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "pvhikjs:")) != -1 )
	{
		switch(opt)
		{
		case 'p':
			// print matrix data.
			print_matrix = 1;
			break;

		case 'v':
			// validation
			validation = 1;
			break;

		case 'h':
		default:
			print_help(argv[0]);
			exit(0);
			break;
		}
	}
}

int main(int argc, char** argv)
{
	int rank, size;
	int NDIM = 4096;
	int fakeArgc = 1;
	float * a, * b, * c, * c_local;
	MPI_Init(&fakeArgc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int rows = NDIM / size;
	MPI_Status status;


	if (rank == 0) {
		int i, j, k = 1;
		MPI_Request reqs[size - 1];
		
		parse_opt( argc, argv );

		a = (float *)malloc(sizeof(float) * NDIM * NDIM);
		b = (float *)malloc(sizeof(float) * NDIM * NDIM);
		c = (float *)malloc(sizeof(float) * NDIM * NDIM);
		c_local = (float *)malloc(sizeof(float) * NDIM * rows);

		printf("%d x %d x %d\n", NDIM, NDIM, NDIM);

		for( i = 0; i < NDIM; i++ )
		{
			for( j = 0; j < NDIM; j++ )
			{
				a[i * NDIM + j] = k;
				b[i * NDIM + j] = k;
				k++;
			}
		}

		timer_start(1);

		if (size > 1) {
			for (i = 1; i < size; i ++) {
				MPI_Isend(a + NDIM * rows * i, NDIM * rows, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &reqs[i - 1]);
			}
		}
		MPI_Bcast(b, NDIM * NDIM, MPI_FLOAT, 0, MPI_COMM_WORLD);

		mat_mul( c_local, a, b, NDIM, rows );

		MPI_Gather(c_local, NDIM * rows, MPI_FLOAT, c, NDIM * rows, MPI_FLOAT, 0, MPI_COMM_WORLD);

		timer_stop(1);

		printf("Time elapsed : %lf sec\n", timer_read(1));

		if( validation )
			check_mat_mul( c, a, b, NDIM );

		if( print_matrix )
		{
			printf("MATRIX A: \n");
			print_mat(a, NDIM);

			printf("MATRIX B: \n");
			print_mat(b, NDIM);

			printf("MATRIX C: \n");
			print_mat(c, NDIM);
		}

	} else {
		a = (float *)malloc(sizeof(float) * NDIM * rows);
		b = (float *)malloc(sizeof(float) * NDIM * NDIM);
		c_local = (float *)malloc(sizeof(float) * NDIM * rows);

		MPI_Recv(a, NDIM * rows, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Bcast(b, NDIM * NDIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		mat_mul(c_local, a, b, NDIM, rows);

		MPI_Gather(c_local, NDIM * rows, MPI_FLOAT, c, NDIM * rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}

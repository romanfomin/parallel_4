#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#include <unistd.h>

#ifdef _OPENMP
	#include "omp.h"
#else
	double omp_get_wtime()
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec + tv.tv_usec / 1000000.0;
	}

	void omp_set_nested(int val)
	{
		(void)val;
	}

	int omp_get_num_procs()
	{
		return 1;
	}
#endif

void print_arr(double *arr, int len)
{
	int i;
	printf("arr=[");
	for (i=0; i<len; i++)
		printf(" %.2f", arr[i]);
	printf("]\n");
}

int get_seed(int i)
{
	return (i + 1) * (i + 1);
}

void fill_arr(double *arr, int len, int left, int right)
{
	int i;
#ifdef _OPENMP
	#pragma omp parallel for private(i) shared(arr, len, left, right)
#endif
	for (i=0; i<len; i++)
	{
		unsigned int seed = get_seed(i);
		arr[i] = rand_r(&seed) / (double)RAND_MAX * (right - left) + left;
	}
}

void apply_m1_func(double *arr, int len)
{
	int i;
#ifdef _OPENMP
	#pragma omp parallel for default(none) private(i) shared(arr, len)
#endif
	for (i=0; i<len; i++)
	{
		arr[i] = 1 / tanh(sqrt(arr[i]));
	}
}

void apply_m2_func(double *arr, int len, double *arr_copy)
{
	int i;
	double prev;
#ifdef _OPENMP
	#pragma omp parallel for default(none) private(i, prev) shared(arr, arr_copy, len)
#endif
	for (i=0; i<len; i++)
	{
		prev = 0;
		if (i > 0)
			prev = arr_copy[i - 1];
		arr[i] = fabs(sin(arr_copy[i] + prev));
	}
}

void apply_merge_func(double *m1, double *m2, int m2_len)
{
	int i;
#ifdef _OPENMP
	#pragma omp parallel for default(none) private(i) shared(m1, m2, m2_len)
#endif
	for (i=0; i<m2_len; i++)
	{
		m2[i] = pow(m1[i], m2[i]);
	}
}

void copy_arr(double *src, int len, double *dst)
{
	int i;
#ifdef _OPENMP
	#pragma omp parallel for default(none) private(i) shared(src, dst, len)
#endif
	for (i=0; i<len; i++)
		dst[i] = src[i];
}

void stupid_sort_part(double *arr, int len)
{
    int i = 0;
    double tmp;

    while (i < len - 1)
    {
        if (arr[i+1] < arr[i])
        {
            tmp = arr[i];
            arr[i] = arr[i+1];
            arr[i+1] = tmp;
            i = 0;
        }
        else i++;
    }
}

double* merge_arr(double *arr1, int len1, double *arr2, int len2)
{
	double *sorted = (double*)malloc(sizeof(double) * (len1 + len2));
	int p1 = 0;
	int p2 = 0;
	int i = 0;

	while (p1 < len1 || p2 < len2) {
		if (p1 < len1 && p2 < len2) {
			if (arr1[p1] <= arr2[p2]) {
				sorted[i] = arr1[p1];
				p1++;
				i++;
				continue;
			}
			if (arr2[p2] < arr1[p1]) {
				sorted[i] = arr2[p2];
				p2++;
				i++;
				continue;
			}
		}
		if (p1 == len1) {
			while (p2 < len2) {
				sorted[i] = arr2[p2];
				i++;
				p2++;
			}
			break;
		}
		if (p2 == len2) {
			while (p1 < len1) {
				sorted[i] = arr1[p1];
				i++;
				p1++;
			}
			break;
		}
	}

	return sorted;
}


double* stupid_sort_on_cpus(double *arr, int len, int l, int r)
{
#ifdef _OPENMP
    int first_part_len = len / 2;
    int second_part_len = len - first_part_len;
    double *first_arr = arr;
    double *second_arr = arr + first_part_len;

    int second_half = l + (r + 1 - l) / 2;
    #pragma omp parallel
	{
		if (omp_get_thread_num() == l) {
			// printf("thread=%d len1=%d\n", omp_get_thread_num(), first_part_len);
			if (r - l == 1)
				stupid_sort_part(first_arr, first_part_len);
			else
				first_arr = stupid_sort_on_cpus(first_arr, first_part_len, l, second_half - 1);
		}
		if (omp_get_thread_num() == second_half) {
			// printf("thread=%d len2=%d\n", omp_get_thread_num(), second_part_len);
			if (r - l == 1)
				stupid_sort_part(second_arr, second_part_len);
			else
				second_arr = stupid_sort_on_cpus(second_arr, second_part_len, second_half, r);
		}
	}
	return merge_arr(first_arr, first_part_len, second_arr, second_part_len);
#else
	stupid_sort_part(arr, len);
	return arr;
#endif
}

double* stupid_sort(double *arr, int len)
{
	return stupid_sort_on_cpus(arr, len, 0, omp_get_num_procs() - 1);
}

double min_not_null(double *arr, int len)
{
	int i;
	double min_val = DBL_MAX;
	for (i=0; i<len; i++)
	{
		if (arr[i] < min_val && arr[i] > 0)
			min_val = arr[i];
	}
	return min_val;
}

double reduce(double *arr, int len)
{
	int i;
	double min_val = min_not_null(arr, len);
	double x = 0;
#ifdef _OPENMP
	#pragma omp parallel for default(none) private(i) shared(arr, len, min_val) reduction(+:x)
#endif
	for (i=0; i<len; i++)
	{
		if ((int)(arr[i] / min_val) % 2 == 0) {
			double sin_val = sin(arr[i]);
			x += sin_val;
		}
	}
	return x;
}

int get_status(int *status)
{
	int val = 0;
#ifdef _OPENMP
	#pragma omp critical
	{
#endif
		val = *status;
#ifdef _OPENMP
	}
#endif
	return val;
}

void print_status(int *status)
{
	int val;

	val = get_status(status);

	while (val < 100) {
		val = get_status(status);

		printf("Current status: %d%%\n", val);
		sleep(1);
	}
}

void do_main(int N, int *status)
{
	int i;
	int A = 729;
	double start, end;

	start = omp_get_wtime();

	for (i=0; i<50; i++) /* 50 экспериментов */
	{
		int m1_len = N, m2_len = N / 2;
		double *m1 = (double*)malloc(sizeof(double) * m1_len);
		double *m2 = (double*)malloc(sizeof(double) * m2_len);
		double *m2_copy = (double*)malloc(sizeof(double) * m2_len);
		// double x;


		// printf("\nFill arrays\n");
		fill_arr(m1, m1_len, 1, A);
		fill_arr(m2, m2_len, A, 10 * A);
		// print_arr(m1, m1_len);
		// print_arr(m2, m2_len);

		// printf("\nMap\n");
		apply_m1_func(m1, m1_len);
		copy_arr(m2, m2_len, m2_copy);
		apply_m2_func(m2, m2_len, m2_copy);
		// print_arr(m1, m1_len);
		// print_arr(m2, m2_len);

		// printf("\nMerge\n");
		apply_merge_func(m1, m2, m2_len);
		// print_arr(m1, m1_len);
		// print_arr(m2, m2_len);

		// printf("\nSort\n");
		m2 = stupid_sort(m2, m2_len);
		// print_arr(m1, m1_len);
		// print_arr(m2, m2_len);

		reduce(m2, m2_len);
		// printf("X=%.4f\n", x);

		free(m1);
		free(m2);
		free(m2_copy);

#ifdef _OPENMP
		#pragma omp critical
		{
#endif
			*status += 2;
#ifdef _OPENMP
		}
#endif
	}

	end = omp_get_wtime();

	printf("%d %.0f\n", N, (end - start) * 1000);
	// printf("%.0f\n", (end - start) * 1000);
}

int main(int argc, char* argv[])
{
	int N = atoi(argv[1]); /* N равен первому параметру командной строки */
	int status = 0;
	omp_set_nested(1);

#ifdef _OPENMP
	#pragma omp parallel sections
	{
		#pragma omp section
		{ 
		    do_main(N, &status);
		}
		#pragma omp section
		{ 
	    	print_status(&status);
		}
	}
#else
	do_main(N, &status);
#endif
	
	return 0;
}

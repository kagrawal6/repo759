// msort.cpp
#include "msort.h"
#include <omp.h>

//helper functions
static void parallelMergeSort(int* arr, int left, int right, std::size_t threshold);
static void merge(int* arr, int left, int mid, int right);
static void insertionSort(int* arr, int left, int right);

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    // parallel region for tasks inside
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            // Sort the entire array from index 0 to n-1
            parallelMergeSort(arr, 0, static_cast<int>(n - 1), threshold);
        }

    }
}


// parallelMergeSort: recursively splits array arr[left..right],
// spawning tasks if subarray_size >= threshold.
// For subarray_size < threshold, fallback to insertionSort a serial sort.

static void parallelMergeSort(int* arr, int left, int right, std::size_t threshold) {
    // base case: 0 or 1 elements => already sorted
    if (left >= right) {
        return;
    }
    int length = right - left + 1;

    // Switch to a serial insertion sort if below threshold
    if (static_cast<std::size_t>(length) < threshold) {
        insertionSort(arr, left, right);
        return;
    }

    // Otherwise, we do parallel tasks
    int mid = left + (length / 2);

    // task for the left half
    #pragma omp task firstprivate(left, mid) shared(arr)
    {
        // Sort [left..(mid-1)]
        parallelMergeSort(arr, left, mid - 1, threshold);
    }

    // task for the right half
    #pragma omp task firstprivate(mid, right) shared(arr)
    {
        // Sort [mid..right]
        parallelMergeSort(arr, mid, right, threshold);
    }

    // Wait for both tasks to complete before merging
    #pragma omp taskwait

    // Merge the two sorted halves
    merge(arr, left, mid, right);
}


// merge: Merges sorted subarrays [left..mid-1] and [mid..right]
//        into arr[left..right] in ascending order.

static void merge(int* arr, int left, int mid, int right) {
    int len1 = mid - left;
    int len2 = right - mid + 1;

    // Temporary buffer for merged data
    int* temp = new int[len1 + len2];

    int idx1 = left;  // index in left subarray
    int idx2 = mid;   // index in right subarray
    int idxTemp = 0;  // index in temp

    // Merge smaller elements first
    while (idx1 < mid && idx2 <= right) {
        if (arr[idx1] <= arr[idx2]) {
            temp[idxTemp++] = arr[idx1++];
        } else {
            temp[idxTemp++] = arr[idx2++];
        }
    }

    // Copy any leftover from left subarray
    while (idx1 < mid) {
        temp[idxTemp++] = arr[idx1++];
    }

    // Copy any leftover from right subarray
    while (idx2 <= right) {
        temp[idxTemp++] = arr[idx2++];
    }

    // Write merged elements back
    for (int i = 0; i < (len1 + len2); i++) {
        arr[left + i] = temp[i];
    }

    delete[] temp;
}


// insertionSort: Serially sorts arr[left..right] in ascending order

static void insertionSort(int* arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
#include <chrono>

using namespace std;

const int INITIAL_CAPACITY = 1000;

struct WordCount {
    string word;
    int count;
};

void resizeWordCounts(WordCount*& arr, int& capacity) {
    if (capacity < 0) {
        return;
    }
    
    int newCapacity = capacity * 2;
    WordCount* newArr = new WordCount[newCapacity];
    for (int i = 0; i < capacity; ++i) {
        newArr[i] = arr[i];
    }
    delete[] arr;
    arr = newArr;
    capacity = newCapacity;
}

char toLower(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c + ('a' - 'A');
    }
    return c;
}

void processTextChunk(const string& textChunk, WordCount*& localWordCounts, int& localWordCountSize, int& localWordCountCapacity) {
    string word;
    for (char c : textChunk) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '-' || c == 39) {
            word += toLower(c);
        } else if (!word.empty()) {
            bool found = false;
            for (int i = 0; i < localWordCountSize; ++i) {
                if (localWordCounts[i].word == word) {
                    localWordCounts[i].count++;
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (localWordCountSize == localWordCountCapacity) {
                    resizeWordCounts(localWordCounts, localWordCountCapacity);
                }
                localWordCounts[localWordCountSize++] = WordCount{word, 1};
            }
            word.clear();
        }
    }
    if (!word.empty()) {
        bool found = false;
        for (int i = 0; i < localWordCountSize; ++i) {
            if (localWordCounts[i].word == word) {
                localWordCounts[i].count++;
                found = true;
                break;
            }
        }
        if (!found) {
            if (localWordCountSize == localWordCountCapacity) {
                resizeWordCounts(localWordCounts, localWordCountCapacity);
            }
            localWordCounts[localWordCountSize++] = WordCount{word, 1};
        }
    }
}

int myPartition(int low, int high, WordCount* arr) {
    WordCount pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j < high; j++) {
        if (arr[j].count > pivot.count) {
            i++;
            // Swap arr[i] and arr[j]
            WordCount temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // Swap arr[i + 1] and arr[high] (or pivot)
    WordCount temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;
}

void seq_qsort(int p, int r, WordCount* arr) {
    if (p < r) {
        int q = myPartition(p, r, arr);
        seq_qsort(p, q - 1, arr);
        seq_qsort(q + 1, r, arr);
    }
}

void q_sort_sections(int p, int r, WordCount* data, int low_limit) {
    if (p < r) {
        if (r - p < low_limit) 
        {
            return seq_qsort(p, r, data);
        }else{
            int q = myPartition(p, r, data);
            // only 2 threads because we only have 2 things to do in this recursive call
            #pragma omp parallel sections shared(data) num_threads(2)
            {
                #pragma omp section
                q_sort_sections(p, q - 1, data, low_limit);
                #pragma omp section
                q_sort_sections(q + 1, r, data, low_limit);
            }
        }
    } 
}

void par_q_sort_sections(int p, int r, WordCount* data){
    q_sort_sections(p, r, data, omp_get_num_threads() - 1);
}

void mergeGlobalCounts(WordCount*& globalCounts, int& globalSize, int& globalCapacity, WordCount* localCounts, int localSize) {
    for (int i = 0; i < localSize; ++i) {
        bool found = false;
        for (int j = 0; j < globalSize; ++j) {
            if (globalCounts[j].word == localCounts[i].word) {
                globalCounts[j].count += localCounts[i].count;
                found = true;
                break;
            }
        }
        if (!found) {
            if (globalSize == globalCapacity) {
                resizeWordCounts(globalCounts, globalCapacity);
            }
            globalCounts[globalSize++] = localCounts[i];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <inputFile> <numberOfThreads>\n";
        return 1;
    }
    int numThreads = atoi(argv[2]);

    string inputFile = argv[1];
    ifstream file(inputFile);
    if (!file) {
        cerr << "Error opening file: " << inputFile << "\n";
        return 1;
    }

    string fileContents((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
  
    WordCount** localWordCounts = new WordCount*[numThreads];
    int* localWordCountSizes = new int[numThreads]{0};
    int* localWordCountCapacities = new int[numThreads];

    for (int i = 0; i < numThreads; ++i) {
        localWordCounts[i] = new WordCount[INITIAL_CAPACITY];
        localWordCountCapacities[i] = INITIAL_CAPACITY;
    }

    size_t chunkSize = fileContents.size() / numThreads;
    string chunks[numThreads];
    auto startTime = chrono::high_resolution_clock::now();

    size_t start, end;
    bool endChanged = false;
    for (size_t i = 0; i < numThreads; ++i) {
        if (endChanged) {
            // adjust the starting index if we had to move the previous ending index
            start = end + 1;
        } else {
            start = i * chunkSize;
        }
        end = (i + 1) * chunkSize;
        if (i == numThreads - 1) {
            end = fileContents.size();
        }
        while (end < fileContents.size() && isalpha(fileContents[end])) {
            ++end;
            endChanged = true;
        }
        chunks[i] = fileContents.substr(start, end - start);
    }


    WordCount* globalWordCounts = new WordCount[INITIAL_CAPACITY];
    int globalWordCountSize = 0;
    int globalWordCountCapacity = INITIAL_CAPACITY;

    omp_set_dynamic(false);
    omp_set_num_threads(numThreads);
    #pragma omp parallel 
    {
        int tNum = omp_get_thread_num();
        processTextChunk(chunks[tNum],ref(localWordCounts[tNum]), ref(localWordCountSizes[tNum]), ref(localWordCountCapacities[tNum]));

        #pragma omp for
        for (int i = 0; i < numThreads; ++i) {
            mergeGlobalCounts(globalWordCounts, globalWordCountSize, globalWordCountCapacity, localWordCounts[i], localWordCountSizes[i]);
        }
    }

    par_q_sort_sections(0, globalWordCountSize, globalWordCounts);

    auto endTime = chrono::high_resolution_clock::now(); 

    chrono::duration<double, milli> runtime = endTime - startTime;
    cout << "Runtime: " << runtime.count() << " ms" << endl;   

    ofstream outputFile("output.txt");
    for (int i = 0; i < globalWordCountSize; ++i) {
        outputFile << globalWordCounts[i].word << " " << globalWordCounts[i].count << "\n";
    }
    outputFile.close();

    for (int i = 0; i < numThreads; ++i) {
        delete[] localWordCounts[i];
    }
    delete[] localWordCounts;
    delete[] localWordCountSizes;
    delete[] localWordCountCapacities;
    delete[] globalWordCounts;

    localWordCountCounts = nullptr;
    localWordCountSizes = nullptr;
    localWordCountCapacities = nullptr;
    globalWordCounts = nullptr;
    
    return 0;
}

#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
#include <chrono>

using namespace std;

const int INITIAL_CAPACITY = 100;

struct WordCount {
    string word;
    int count;
};

void resizeWordCounts(WordCount*& arr, int& capacity) {
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

void mergeGlobalCounts(WordCount*& globalCounts, int& globalSize, int& globalCapacity, WordCount* localCounts, int localSize) {
    #pragma omp critical
    {
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
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <inputFile> <numberOfThreads>\n";
        return 1;
    }

    string inputFile = argv[1];
    ifstream file(inputFile);
    if (!file) {
        cerr << "Error opening file: " << inputFile << "\n";
        return 1;
    }

    string fileContents((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    int numThreads = atoi(argv[2]);
    omp_set_num_threads(numThreads);

    WordCount* globalWordCounts = new WordCount[INITIAL_CAPACITY];
    int globalWordCountSize = 0;
    int globalWordCountCapacity = INITIAL_CAPACITY;

    #pragma omp parallel
    {
        WordCount* localWordCounts = new WordCount[INITIAL_CAPACITY];
        int localWordCountSize = 0;
        int localWordCountCapacity = INITIAL_CAPACITY;

        #pragma omp for nowait
        for (int i = 0; i < numThreads; ++i) {
            size_t start = i * fileContents.size() / numThreads;
            size_t end = (i + 1) * fileContents.size() / numThreads;
            if (i == numThreads - 1) {
                end = fileContents.size();
            }
            string chunk = fileContents.substr(start, end - start);
            processTextChunk(chunk, localWordCounts, localWordCountSize, localWordCountCapacity);
        }

        mergeGlobalCounts(globalWordCounts, globalWordCountSize, globalWordCountCapacity, localWordCounts, localWordCountSize);
        
        delete[] localWordCounts;
    }

    // Sorting globalWordCounts if necessary

    auto endTime = chrono::high_resolution_clock::now();

    // Output results and runtime...

    delete[] globalWordCounts;
    return 0;
}

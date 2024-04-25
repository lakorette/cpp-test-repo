#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

const int MAX_RESULT_DOCUMENT_COUNT = 5;
const double EPSILON = 1e-6;

string ReadLine() 
{
    string s;
    getline(cin, s);
    return s;
}

int ReadLineWithNumber() 
{
    int result = 0;
    cin >> result;
    ReadLine();
    return result;
}

vector<string> SplitIntoWords(const string& text) 
{
    vector<string> words;
    string word;
    for (const char c : text) 
    {
        if (c == ' ') {
            if (!word.empty()) 
            {
                words.push_back(word);
                word.clear();
            }
        } 
        else 
        {
            word += c;
        }
    }
    if (!word.empty()) 
    {
        words.push_back(word);
    }

    return words;
}

vector<int> ReadLineWithRatings()
{
    int number_of_ratings;
    cin >> number_of_ratings;
    vector<int> ratings(number_of_ratings);
    for (int i = 0; i < number_of_ratings; ++i)
    {
        cin >> ratings[i];
    }
    ReadLine();
    return ratings;
}

struct Document {
    int id;
    double relevance;
    int rating;
};

enum class DocumentStatus
{
    ACTUAL,
    IRRELEVANT,
    BANNED,
    REMOVED
};

class SearchServer {
public:
    void SetStopWords(const string& text) 
    {
        for (const string& word : SplitIntoWords(text)) 
        {
            stop_words_.insert(word);
        }
    }

    void AddDocument(int document_id, const string& document, 
        DocumentStatus status, const vector<int> ratings) 
    {
        const vector<string> words = SplitIntoWordsNoStop(document);
        const double inverse_vector_size = 1. / words.size();

        ++document_count_;
        
        for (const string& word : words)
        {
            documents_to_info_[document_id].words.insert(word);
            documents_to_info_[document_id].status = status;
            words_to_documents_[word].insert(document_id);
            words_to_frequencies_[word][document_id] += inverse_vector_size;
        }

        documents_to_info_[document_id].rating = ComputeAverageRating(ratings);
    }

    template <typename Func>
    vector<Document> FindTopDocuments(const string& raw_query, 
        Func predicate) const 
    {
        const Query query = ParseQuery(raw_query);
        vector<Document> matched_documents = FindAllDocuments(query, predicate);

        sort(matched_documents.begin(), matched_documents.end(),
             [](const Document& lhs, const Document& rhs) 
             {
                 return lhs.relevance > rhs.relevance || 
                    (std::abs(lhs.relevance - rhs.relevance) < EPSILON && 
                    lhs.rating > rhs.rating);
             });
        if (matched_documents.size() > MAX_RESULT_DOCUMENT_COUNT) 
        {
            matched_documents.resize(MAX_RESULT_DOCUMENT_COUNT);
        }
        return matched_documents;
    }

    vector<Document> FindTopDocuments(const string& raw_query, 
                                DocumentStatus document_status) const 
    {
        return FindTopDocuments(raw_query, [document_status]
                (int id, DocumentStatus status, int rating)
                {
                    return status == document_status;
                });
    }

    vector<Document> FindTopDocuments(const string& raw_query) const 
    {
        return FindTopDocuments(raw_query, [](int id, DocumentStatus status, 
                                            int rating)
                            {
                                return status == DocumentStatus::ACTUAL;
                            });
    }

    tuple<vector<string>, DocumentStatus> MatchDocument(
        const string& raw_query, int document_id) const
    {
        Query query = ParseQuery(raw_query);
        vector<string> matched_plus_words;

        for (const string& word : query.minus_words)
        {
            if (documents_to_info_.at(document_id).words.count(word) > 0)
            return tuple{matched_plus_words, 
                documents_to_info_.at(document_id).status};
        }

        for (const string& word : query.plus_words)
        {
            if (documents_to_info_.at(document_id).words.count(word) > 0)
            {
                matched_plus_words.push_back(word);
            }
        }
        return tuple(matched_plus_words, 
            documents_to_info_.at(document_id).status);
    } 

    int GetDocumentCount() const
    {
        return documents_to_info_.size();
    }

private:
    struct DocumentInformation
    {
        set<string> words;
        DocumentStatus status;
        int rating;
    };

    int document_count_ = 0;

    map<string, set<int>> words_to_documents_;
    map<int, DocumentInformation> documents_to_info_;
    map<string, map<int, double>> words_to_frequencies_;
    set<string> stop_words_;

    struct Query
    {
        set<string> minus_words;
        set<string> plus_words;
    };
    
    bool IsStopWord(const string& word) const 
    {
        return stop_words_.count(word) > 0;
    }

    vector<string> SplitIntoWordsNoStop(const string& text) const 
    {
        vector<string> words;
        for (const string& word : SplitIntoWords(text)) 
        {
            if (!IsStopWord(word)) 
            {
                words.push_back(word);
            }
        }
        return words;
    }

    Query ParseQuery(const string& text) const 
    {
        set<string> query_words;
        Query query;
        for (const string& word : SplitIntoWordsNoStop(text)) 
        {
            query_words.insert(word);
        }
        
        for (const string& word : query_words)
        {
            if (word.empty())
            {
                continue;
            }
            if (word.front() == '-')
            {
                query.minus_words.insert(
                    accumulate(word.begin() + 1, word.end(), ""s));
            }
            else
            {
                query.plus_words.insert(word);
            }
        }

        for (const string& word : query.minus_words)
        {
            if (query.plus_words.count(word) > 0)
            {
                query.plus_words.erase(word);
            }
        }

        return query;
    }

    template <typename Func>
    vector<Document> FindAllDocuments(const Query& query, Func predicate) const 
    {
        vector<Document> matched_documents;

        map<int, double> ids_to_relevance;

        for (const string& word : query.plus_words)
        {
            if (query.minus_words.count(word) > 0)
            {
                continue;
            }
            if (words_to_documents_.count(word) == 0)
            {
                continue;
            }
            const double inverse_document_frequency = 
                ComputeInverseDocumentFrequency(word);
            for (int id : words_to_documents_.at(word))
            {
                bool allowed_document = true;
                for (const string& minus_word : query.minus_words)
                {
                    if (documents_to_info_.at(id).words.count(minus_word) > 0)
                    {
                        allowed_document = false;
                    }
                }
                if (!(allowed_document && predicate(id, 
                                documents_to_info_.at(id).status, 
                                documents_to_info_.at(id).rating)))
                {
                    continue;
                }
                ids_to_relevance[id] += 
                    inverse_document_frequency * 
                    words_to_frequencies_.at(word).at(id);
            }
        }
        matched_documents.reserve(ids_to_relevance.size());
        for (const auto& [id, relevance] : ids_to_relevance)
        {
            matched_documents.push_back(
                Document{id, relevance, documents_to_info_.at(id).rating});
        }
        return matched_documents;
    }

    double ComputeInverseDocumentFrequency(const string& word) const 
    {
        return log(static_cast<double>(document_count_) / 
            words_to_documents_.at(word).size());
    }

    static int ComputeAverageRating(const vector<int>& ratings)
    {
        if (ratings.size() == 0)
        {
            return 0;
        }
        return accumulate(ratings.begin(), ratings.end(), 0) / 
            static_cast<int>(ratings.size());
    }
};

void PrintDocument(const Document& document) {
    cout << "{ "s
         << "document_id = "s << document.id << ", "s
         << "relevance = "s << document.relevance << ", "s
         << "rating = "s << document.rating
         << " }"s << endl;
}

int main() {
    SearchServer search_server;
    search_server.SetStopWords("и в на"s);
    search_server.AddDocument(0, "белый кот и модный ошейник"s,        DocumentStatus::ACTUAL, {8, -3});
    search_server.AddDocument(1, "пушистый кот пушистый хвост"s,       DocumentStatus::ACTUAL, {7, 2, 7});
    search_server.AddDocument(2, "ухоженный пёс выразительные глаза"s, DocumentStatus::ACTUAL, {5, -12, 2, 1});
    search_server.AddDocument(3, "ухоженный скворец евгений"s,         DocumentStatus::BANNED, {9});
    cout << "ACTUAL by default:"s << endl;
    for (const Document& document : search_server.FindTopDocuments("пушистый ухоженный кот"s)) {
        PrintDocument(document);
    }
    cout << "BANNED:"s << endl;
    for (const Document& document : search_server.FindTopDocuments("пушистый ухоженный кот"s, DocumentStatus::BANNED)) {
        PrintDocument(document);
    }
    cout << "Even ids:"s << endl;
    for (const Document& document : search_server.FindTopDocuments("пушистый ухоженный кот"s, [](int document_id, DocumentStatus status, int rating) { return document_id % 2 == 0; })) {
        PrintDocument(document);
    }
    return 0;
}
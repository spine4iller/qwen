using RagApp.Data;
using RagApp.Services;

namespace RagApp;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("=== RAG Приложение на .NET с Ollama ===\n");
        
        // Парсинг аргументов командной строки
        string ollamaBaseUrl = "http://localhost:11434";
        string embeddingModel = "nomic-embed-text";
        string chatModel = "gemma3:1b";//"llama3.2";
        
        Console.WriteLine($"Ollama URL: {ollamaBaseUrl}");
        Console.WriteLine($"Модель эмбеддингов: {embeddingModel}");
        Console.WriteLine($"Чат-модель: {chatModel}\n");
        
        // Инициализация сервисов с авто-определением доступности Ollama
        var embeddingService = await OllamaServiceFactory.CreateEmbeddingServiceAsync(ollamaBaseUrl, embeddingModel);
        var llmService = await OllamaServiceFactory.CreateLlmServiceAsync(ollamaBaseUrl, chatModel);
        var vectorStore = new InMemoryVectorStore();
        var ragService = new RagService(embeddingService, vectorStore, llmService);
        
        // Индексация документов
        Console.WriteLine("\nИндексация документов...");
        
        // Вариант 1: Индексация большого документа с автоматическим разбиением на чанки
        var fullDocument = File.ReadAllText(@"c:\Games\devilutionx\README.txt");
        // "Мишка серого цвета";//SampleData.GetSampleDocuments();
        await ragService.IndexDocumentAsync(fullDocument, "documentation.txt");
        
        // Вариант 2: Индексация отдельных чанков
     /*   var sampleChunks = SampleData.GetSampleChunks();
        foreach (var (text, source) in sampleChunks)
        {
            var embedding = await embeddingService.GenerateEmbeddingAsync(text);
            var chunk = new Models.DocumentChunk
            {
                Content = text,
                Source = source,
                Embedding = embedding
            };
            await vectorStore.AddDocumentAsync(chunk);
        }
        */
        Console.WriteLine($"Документы проиндексированы!\n");
        
        // Демонстрация работы RAG
        var questions = new List<string>
        {
           "о чём документ?", "какой порт надо открыть для игры по сети?"
        };
        
        foreach (var question in questions)
        {
            Console.WriteLine(new string('-', 50));
            Console.WriteLine($"Вопрос: {question}\n");
            
            var result = await ragService.QueryAsync(question);
            
            Console.WriteLine("Ответ:");
            Console.WriteLine(result.Answer);
            
            Console.WriteLine($"\nНайдено релевантных чанков: {result.RelevantChunks.Count}");
            foreach (var chunk in result.RelevantChunks)
            {
                Console.WriteLine($"  - Источник: {chunk.Source}");
                Console.WriteLine($"    Текст: {chunk.Content[..Math.Min(80, chunk.Content.Length)]}...");
            }
            Console.WriteLine();
        }
        
        Console.WriteLine("До свидания!");
    }
}

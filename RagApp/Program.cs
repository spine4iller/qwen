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
        string chatModel = "llama3.2";
        
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--ollama-url":
                    if (i + 1 < args.Length) ollamaBaseUrl = args[++i];
                    break;
                case "--embedding-model":
                    if (i + 1 < args.Length) embeddingModel = args[++i];
                    break;
                case "--chat-model":
                    if (i + 1 < args.Length) chatModel = args[++i];
                    break;
                case "--help":
                    PrintHelp();
                    return;
            }
        }
        
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
        var fullDocument = SampleData.GetSampleDocuments();
        await ragService.IndexDocumentAsync(fullDocument, "documentation.txt");
        
        // Вариант 2: Индексация отдельных чанков
        var sampleChunks = SampleData.GetSampleChunks();
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
        
        Console.WriteLine($"Документы проиндексированы!\n");
        
        // Демонстрация работы RAG
        var questions = new List<string>
        {
            "Что такое .NET?",
            "Какой язык программирования используется в ASP.NET Core?",
            "Что такое Azure?"
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
        
        // Интерактивный режим
        Console.WriteLine(new string('-', 50));
        Console.WriteLine("Интерактивный режим (введите 'exit' для выхода):\n");
        
        while (true)
        {
            Console.Write("Ваш вопрос: ");
            var input = Console.ReadLine();
            
            if (string.IsNullOrWhiteSpace(input) || input.ToLower() == "exit")
                break;
            
            var result = await ragService.QueryAsync(input);
            
            Console.WriteLine("\nОтвет:");
            Console.WriteLine(result.Answer);
            
            if (result.RelevantChunks.Any())
            {
                Console.WriteLine("\nИсточники:");
                foreach (var chunk in result.RelevantChunks)
                {
                    Console.WriteLine($"  • {chunk.Source}: {chunk.Content[..Math.Min(60, chunk.Content.Length)]}...");
                }
            }
            Console.WriteLine();
        }
        
        Console.WriteLine("До свидания!");
    }
    
    static void PrintHelp()
    {
        Console.WriteLine("""
        RAG Приложение с поддержкой Ollama
        
        Использование:
          dotnet run [--ollama-url <URL>] [--embedding-model <MODEL>] [--chat-model <MODEL>]
        
        Параметры:
          --ollama-url       URL Ollama сервера (по умолчанию: http://localhost:11434)
          --embedding-model  Модель для эмбеддингов (по умолчанию: nomic-embed-text)
          --chat-model       Модель для генерации ответов (по умолчанию: llama3.2)
          --help             Показать эту справку
        
        Примеры:
          dotnet run
          dotnet run --ollama-url http://localhost:11434 --chat-model mistral
          dotnet run --embedding-model all-minilm --chat-model llama3.2
        
        Для работы с Ollama:
          1. Установите Ollama: https://ollama.ai
          2. Запустите сервер: ollama serve
          3. Скачайте модели:
             ollama pull nomic-embed-text
             ollama pull llama3.2
        """);
    }
}

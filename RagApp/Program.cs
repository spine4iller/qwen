using RagApp.Data;
using RagApp.Services;

namespace RagApp;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("=== RAG Приложение на .NET ===\n");
        
        // Инициализация сервисов
        var embeddingService = new SimpleEmbeddingService();
        var vectorStore = new InMemoryVectorStore();
        var llmService = new SimpleLlmService();
        var ragService = new RagService(embeddingService, vectorStore, llmService);
        
        // Индексация документов
        Console.WriteLine("Индексация документов...");
        
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
}

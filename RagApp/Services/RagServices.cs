using System.Numerics.Tensors;
using RagApp.Models;

namespace RagApp.Services;

public interface IEmbeddingService
{
    Task<float[]> GenerateEmbeddingAsync(string text);
}

public class SimpleEmbeddingService : IEmbeddingService
{
    // Простая эмуляция эмбеддингов для демонстрации
    // В реальном проекте используйте Azure OpenAI или другую модель
    public Task<float[]> GenerateEmbeddingAsync(string text)
    {
        // Генерируем псевдо-эмбеддинг на основе хеша текста
        var hash = text.GetHashCode();
        var embedding = new float[1536]; // Размерность как у text-embedding-ada-002
        
        Random rng = new Random(hash);
        for (int i = 0; i < embedding.Length; i++)
        {
            embedding[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        
        // Нормализуем вектор
        float magnitude = 0;
        foreach (var value in embedding)
        {
            magnitude += value * value;
        }
        magnitude = (float)Math.Sqrt(magnitude);
        
        if (magnitude > 0)
        {
            for (int i = 0; i < embedding.Length; i++)
            {
                embedding[i] /= magnitude;
            }
        }
        
        return Task.FromResult(embedding);
    }
}

public interface IVectorStore
{
    Task AddDocumentAsync(DocumentChunk chunk);
    Task<List<DocumentChunk>> SearchSimilarAsync(float[] queryEmbedding, int topK = 3);
}

public class InMemoryVectorStore : IVectorStore
{
    private readonly List<DocumentChunk> _chunks = new();
    
    public Task AddDocumentAsync(DocumentChunk chunk)
    {
        _chunks.Add(chunk);
        return Task.CompletedTask;
    }
    
    public Task<List<DocumentChunk>> SearchSimilarAsync(float[] queryEmbedding, int topK = 3)
    {
        var similarities = _chunks.Select(chunk => new
        {
            Chunk = chunk,
            Similarity = CosineSimilarity(queryEmbedding, chunk.Embedding)
        })
        .OrderByDescending(x => x.Similarity)
        .Take(topK)
        .Select(x => x.Chunk)
        .ToList();
        
        return Task.FromResult(similarities);
    }
    
    private static float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length) return 0;
        
        float dot = 0;
        float normA = 0;
        float normB = 0;
        
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0 || normB == 0) return 0;
        
        return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
    }
}

public interface ILlmService
{
    Task<string> GenerateAnswerAsync(string question, string context);
}

public class SimpleLlmService : ILlmService
{
    // Заглушка для LLM
    // В реальном проекте используйте Azure OpenAI API
    public Task<string> GenerateAnswerAsync(string question, string context)
    {
        var response = $"""
        На основе предоставленного контекста:
        
        {context}
        
        Ответ на вопрос "{question}":
        
        Это демонстрационный ответ. Для получения реальных ответов подключите Azure OpenAI API 
        или другой LLM сервис в классе SimpleLlmService.
        
        Контекст содержит {context.Length} символов справочной информации.
        """;
        
        return Task.FromResult(response);
    }
}

public class RagService
{
    private readonly IEmbeddingService _embeddingService;
    private readonly IVectorStore _vectorStore;
    private readonly ILlmService _llmService;
    
    public RagService(IEmbeddingService embeddingService, IVectorStore vectorStore, ILlmService llmService)
    {
        _embeddingService = embeddingService;
        _vectorStore = vectorStore;
        _llmService = llmService;
    }
    
    public async Task IndexDocumentAsync(string text, string source = "unknown")
    {
        // Разбиваем текст на чанки (простое разбиение по предложениям)
        var chunks = SplitIntoChunks(text, 200);
        
        foreach (var chunkText in chunks)
        {
            var embedding = await _embeddingService.GenerateEmbeddingAsync(chunkText);
            var chunk = new DocumentChunk
            {
                Content = chunkText,
                Source = source,
                Embedding = embedding
            };
            
            await _vectorStore.AddDocumentAsync(chunk);
        }
    }
    
    public async Task<QueryResult> QueryAsync(string question)
    {
        // Генерируем эмбеддинг для вопроса
        var queryEmbedding = await _embeddingService.GenerateEmbeddingAsync(question);
        
        // Ищем похожие документы
        var relevantChunks = await _vectorStore.SearchSimilarAsync(queryEmbedding, topK: 3);
        
        if (!relevantChunks.Any())
        {
            return new QueryResult
            {
                Answer = "Не найдено релевантной информации в базе знаний.",
                RelevantChunks = new List<DocumentChunk>()
            };
        }
        
        // Формируем контекст из найденных чанков
        var context = string.Join("\n\n", relevantChunks.Select(c => c.Content));
        
        // Генерируем ответ с помощью LLM
        var answer = await _llmService.GenerateAnswerAsync(question, context);
        
        return new QueryResult
        {
            Answer = answer,
            RelevantChunks = relevantChunks
        };
    }
    
    private static List<string> SplitIntoChunks(string text, int maxChunkSize)
    {
        var chunks = new List<string>();
        
        // Простое разбиение по предложениям
        var sentences = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        
        var currentChunk = new StringBuilder();
        
        foreach (var sentence in sentences)
        {
            var trimmedSentence = sentence.Trim();
            if (trimmedSentence.Length == 0) continue;
            
            if (currentChunk.Length + trimmedSentence.Length + 1 > maxChunkSize && currentChunk.Length > 0)
            {
                chunks.Add(currentChunk.ToString().TrimEnd() + ".");
                currentChunk.Clear();
            }
            
            currentChunk.Append(trimmedSentence).Append(". ");
        }
        
        if (currentChunk.Length > 0)
        {
            chunks.Add(currentChunk.ToString().TrimEnd() + ".");
        }
        
        return chunks;
    }
}

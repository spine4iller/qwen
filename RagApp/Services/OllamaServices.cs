using System.Net.Http;
using System.Text;
using System.Text.Json;
using RagApp.Models;

namespace RagApp.Services;

/// <summary>
/// Сервис для работы с локальной Ollama моделью
/// Поддерживает генерацию эмбеддингов и ответов через Ollama API
/// </summary>
public class OllamaEmbeddingService : IEmbeddingService
{
    private readonly HttpClient _httpClient;
    private readonly string _embeddingModel;

    public OllamaEmbeddingService(string baseUrl = "http://localhost:11434", string embeddingModel = "nomic-embed-text")
    {
        _httpClient = new HttpClient();
        _httpClient.BaseAddress = new Uri(baseUrl);
        _httpClient.Timeout = TimeSpan.FromMinutes(2); // Эмбеддинги могут занимать время
        _embeddingModel = embeddingModel;
    }

    public async Task<float[]> GenerateEmbeddingAsync(string text)
    {
        var requestBody = new
        {
            model = _embeddingModel,
            prompt = text
        };

        var json = JsonSerializer.Serialize(requestBody);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("/api/embeddings", content);
        
        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync();
            throw new InvalidOperationException($"Ollama API error: {response.StatusCode} - {error}");
        }

        var responseJson = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(responseJson);
        
        if (doc.RootElement.TryGetProperty("embedding", out var embeddingElement))
        {
            var embeddings = new List<float>();
            foreach (var element in embeddingElement.EnumerateArray())
            {
                embeddings.Add(element.GetSingle());
            }
            return embeddings.ToArray();
        }

        throw new InvalidOperationException("No embedding found in Ollama response");
    }
}

/// <summary>
/// Сервис для генерации ответов через локальную Ollama модель
/// </summary>
public class OllamaLlmService : ILlmService
{
    private readonly HttpClient _httpClient;
    private readonly string _chatModel;

    public OllamaLlmService(string baseUrl = "http://localhost:11434", string chatModel = "llama3.2")
    {
        _httpClient = new HttpClient();
        _httpClient.BaseAddress = new Uri(baseUrl);
        _httpClient.Timeout = TimeSpan.FromMinutes(5); // LLM ответы могут занимать время
        _chatModel = chatModel;
    }

    public async Task<string> GenerateAnswerAsync(string question, string context)
    {
        var systemPrompt = """
            Ты полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.
            Если ответ не может быть найден в контексте, честно скажи об этом.
            Отвечай на том же языке, на котором задан вопрос.
            Будь краток и точен.
            """;

        var userPrompt = $"""
            Контекст:
            {context}

            Вопрос: {question}

            Ответ:
            """;

        var requestBody = new
        {
            model = _chatModel,
            messages = new[]
            {
                new { role = "system", content = systemPrompt },
                new { role = "user", content = userPrompt }
            },
            stream = false,
            options = new
            {
                temperature = 0.7,
                top_p = 0.9
            }
        };

        var json = JsonSerializer.Serialize(requestBody, new JsonSerializerOptions 
        { 
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase 
        });
        
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("/api/chat", content);
        
        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync();
            throw new InvalidOperationException($"Ollama API error: {response.StatusCode} - {error}");
        }

        var responseJson = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(responseJson);
        
        if (doc.RootElement.TryGetProperty("message", out var messageElement) &&
            messageElement.TryGetProperty("content", out var contentElement))
        {
            return contentElement.GetString() ?? "Пустой ответ от модели";
        }

        throw new InvalidOperationException("No message found in Ollama response");
    }
}

/// <summary>
/// Фабрика для создания сервисов Ollama или заглушек в зависимости от доступности Ollama
/// </summary>
public static class OllamaServiceFactory
{
    /// <summary>
    /// Проверяет доступность Ollama сервера
    /// </summary>
    public static async Task<bool> IsOllamaAvailableAsync(string baseUrl = "http://localhost:11434")
    {
        try
        {
            using var httpClient = new HttpClient();
            httpClient.BaseAddress = new Uri(baseUrl);
            httpClient.Timeout = TimeSpan.FromSeconds(5);
            
            var response = await httpClient.GetAsync("/api/tags");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Создает сервис эмбеддингов: Ollama если доступен, иначе заглушка
    /// </summary>
    public static async Task<IEmbeddingService> CreateEmbeddingServiceAsync(
        string baseUrl = "http://localhost:11434", 
        string embeddingModel = "nomic-embed-text")
    {
        if (await IsOllamaAvailableAsync(baseUrl))
        {
            Console.WriteLine($"✓ Подключен Ollama Embedding Service (модель: {embeddingModel})");
            return new OllamaEmbeddingService(baseUrl, embeddingModel);
        }
        else
        {
            Console.WriteLine("⚠ Ollama недоступен, используется эмуляция эмбеддингов");
            Console.WriteLine("  Для использования реальных эмбеддингов запустите Ollama: ollama serve");
            Console.WriteLine($"  Рекомендуемая модель: ollama pull {embeddingModel}");
            throw new InvalidOperationException("not connected");
        }
    }

    /// <summary>
    /// Создает LLM сервис: Ollama если доступен, иначе заглушка
    /// </summary>
    public static async Task<ILlmService> CreateLlmServiceAsync(
        string baseUrl = "http://localhost:11434", 
        string chatModel = "llama3.2")
    {
        if (await IsOllamaAvailableAsync(baseUrl))
        {
            Console.WriteLine($"✓ Подключен Ollama LLM Service (модель: {chatModel})");
            return new OllamaLlmService(baseUrl, chatModel);
        }
        else
        {
            Console.WriteLine("⚠ Ollama недоступен, используется демонстрационный режим");
            Console.WriteLine("  Для использования реальной LLM запустите Ollama: ollama serve");
            Console.WriteLine($"  Рекомендуемая модель: ollama pull {chatModel}");
            return new SimpleLlmService();
        }
    }
}

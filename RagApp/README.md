# RAG Приложение на .NET

Это демонстрационное приложение реализует архитектуру **RAG (Retrieval-Augmented Generation)** на платформе .NET 8.

## Архитектура

Приложение состоит из следующих компонентов:

### 1. **Модели** (`Models/`)
- `DocumentChunk` - представляет фрагмент документа с эмбеддингом
- `QueryResult` - результат поиска с ответом и релевантными чанками

### 2. **Сервисы** (`Services/`)
- `IEmbeddingService` / `SimpleEmbeddingService` - генерация векторных представлений текста
- `IVectorStore` / `InMemoryVectorStore` - хранение и поиск по векторам
- `ILlmService` / `SimpleLlmService` - генерация ответов через LLM
- `RagService` - основной сервис, объединяющий все компоненты

### 3. **Данные** (`Data/`)
- `SampleData` - пример документов для индексации

## Как это работает

1. **Индексация**: Документы разбиваются на чанки, для каждого генерируется эмбеддинг
2. **Поиск**: Для вопроса генерируется эмбеддинг, находятся похожие чанки (косинусное сходство)
3. **Генерация**: Найденный контекст передается в LLM для формирования ответа

## Запуск

```bash
cd RagApp
dotnet restore
dotnet run
```

## Интеграция с реальными сервисами

Для production-использования замените заглушки на реальные сервисы:

### Azure OpenAI для эмбеддингов:
```csharp
public class AzureEmbeddingService : IEmbeddingService
{
    private readonly OpenAIClient _client;
    
    public async Task<float[]> GenerateEmbeddingAsync(string text)
    {
        var response = await _client.GetEmbeddingsAsync(
            new EmbeddingsOptions("text-embedding-ada-002", new[] { text }));
        
        return response.Value.Data[0].Embedding.ToArray();
    }
}
```

### Azure OpenAI для генерации ответов:
```csharp
public class AzureLlmService : ILlmService
{
    private readonly ChatClient _chatClient;
    
    public async Task<string> GenerateAnswerAsync(string question, string context)
    {
        var prompt = $"""
        Используй следующий контекст для ответа на вопрос.
        
        Контекст:
        {context}
        
        Вопрос: {question}
        
        Ответ:""";
        
        var response = await _chatClient.CompleteAsync(prompt);
        return response.Choices[0].Message.Content;
    }
}
```

## Структура проекта

```
RagApp/
├── Models/
│   └── DocumentChunk.cs
├── Services/
│   └── RagServices.cs
├── Data/
│   └── SampleData.cs
├── Program.cs
└── RagApp.csproj
```

## Расширение

Вы можете добавить:
- Постоянное хранилище векторов (Qdrant, Pinecone, Weaviate)
- Более сложную логику разбиения документов
- Кэширование эмбеддингов
- API endpoint'ы через ASP.NET Core
- Web UI для взаимодействия

# Model Context Protocol (MCP)

هذا المجلد يحتوي مثالًا عمليًا على استخدام **Model Context Protocol** مع Python و LangChain. الفكرة الأساسية أن نعرّف أدوات خارجية في سيرفرات MCP، ثم نجعل الكلاينت يجلب هذه الأدوات ويمررها إلى Agent يستطيع استخدامها أثناء الإجابة.

## مقدمة

تطبيقات الذكاء الاصطناعي لا تحتاج فقط إلى نموذج لغوي، بل تحتاج غالبًا إلى سياق وأدوات: قراءة ملفات، استدعاء API، تنفيذ حسابات، البحث في قاعدة بيانات، أو جلب بيانات من خدمة خارجية. هنا يأتي دور MCP.

يمكن التفكير في MCP كطبقة قياسية بين:

- **MCP Client**: التطبيق أو الـ Agent الذي يريد استخدام أدوات خارجية.
- **MCP Server**: برنامج يعرّف أدوات أو موارد أو Prompts يمكن للعميل اكتشافها واستدعاؤها.
- **Transport**: طريقة نقل رسائل البروتوكول بين العميل والسيرفر، مثل `stdio` أو `streamable-http`.

في هذا المشروع لدينا:

- `weather.py`: سيرفر MCP يقدم أداة `get_weather`.
- `mathservr.py`: سيرفر MCP يقدم أدوات حسابية: `add` و `subtract` و `multiply`.
- `client.py`: كلاينت يجمع أدوات السيرفرين ويعطيها إلى LangChain Agent يستخدم نموذج Groq.

## الشرح النظري لل MCP

MCP هو بروتوكول مفتوح يهدف إلى توحيد طريقة ربط النماذج اللغوية بالأنظمة الخارجية. بدل أن يكتب كل تطبيق طريقة خاصة لاستدعاء الأدوات، يوفر MCP شكلًا موحدًا لاكتشاف القدرات واستدعائها.

المكوّنات المهمة في البروتوكول:

- **Base Protocol**: الرسائل الأساسية مبنية على JSON-RPC 2.0.
- **Lifecycle**: يبدأ الاتصال بمرحلة تهيئة `initialize` يتم فيها الاتفاق على نسخة البروتوكول والقدرات.
- **Server Features**: السيرفر يمكن أن يعرّف أدوات `tools` أو موارد `resources` أو قوالب prompts.
- **Client Features**: الكلاينت يمكن أن يعلن قدرات مثل roots أو sampling حسب التطبيق.

في هذا المثال نستخدم أهم جزء للمبتدئين: **Tools**. السيرفر يعرّف دوال Python كأدوات، والكلاينت يطلب قائمة الأدوات ثم يستدعي الأداة المناسبة حسب سؤال المستخدم.

## ما هو البروتوكول؟

البروتوكول هو اتفاق على شكل الرسائل ومعناها. MCP يستخدم JSON-RPC 2.0، وهذا يعني أن الرسائل تكون غالبًا من ثلاثة أنواع:

### Request

رسالة تطلب تنفيذ عملية، مثل طلب قائمة الأدوات:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

### Response

رسالة رد على طلب سابق، وتحمل نفس `id`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": []
  }
}
```

### Notification

رسالة باتجاه واحد لا تنتظر ردًا، مثل إشعار أن التهيئة اكتملت:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

عند استدعاء أداة، يرسل الكلاينت طلبًا مثل:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "New York"
    }
  }
}
```

## بروتوكولات التخاطب

كلمة transport تعني طريقة نقل رسائل MCP بين العميل والسيرفر. الرسائل نفسها تبقى MCP/JSON-RPC، لكن الطريق الذي تمر منه يختلف.

### 1. stdio

في `stdio` يشغّل الكلاينت السيرفر كعملية فرعية، ثم يتواصل معه عبر:

- `stdin`: الكلاينت يرسل رسائل JSON-RPC إلى السيرفر.
- `stdout`: السيرفر يرد برسائل JSON-RPC إلى الكلاينت.
- `stderr`: يستخدم عادة للـ logs.

هذا مناسب للأدوات المحلية البسيطة. في مشروعنا `mathservr.py` يعمل بهذه الطريقة:

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

وفي `client.py` يتم تشغيله هكذا:

```python
"Math": {
    "transport": "stdio",
    "command": "python",
    "args": ["mathservr.py"],
}
```

### 2. Streamable HTTP

في `streamable-http` يعمل السيرفر كخدمة HTTP مستقلة، ويكون له endpoint واحد خاص بـ MCP. في FastMCP المسار الافتراضي هو:

```text
http://localhost:8000/mcp
```

هذا ليس REST API عاديًا، لذلك فتح `/mcp` في المتصفح قد يعطي `406 Not Acceptable`، وفتح `/` أو `/weather` قد يعطي `404 Not Found`. هذا طبيعي، لأن endpoint يتوقع رسائل MCP من كلاينت MCP، وليس صفحة HTML.

في مشروعنا `weather.py` يعمل بهذه الطريقة:

```python
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

وفي `client.py` يتم الاتصال به هكذا:

```python
"Weather": {
    "transport": "streamable-http",
    "url": "http://localhost:8000/mcp",
}
```

### 3. SSE القديم

بعض المكتبات ما زالت تدعم `sse` للتوافق مع الإصدارات القديمة، لكن في المواصفات الأحدث تم استبدال HTTP+SSE القديم بـ `Streamable HTTP`. لذلك في المشاريع الجديدة يفضّل استخدام `streamable-http` عند الحاجة إلى HTTP.

## شرح كود السيرفر

### weather.py

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The current weather in {location} is sunny with a high of 25°C."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

شرح الأجزاء:

- `FastMCP("Weather")`: ينشئ سيرفر MCP باسم Weather.
- `@mcp.tool()`: يحوّل الدالة إلى أداة قابلة للاكتشاف والاستدعاء من الكلاينت.
- `location: str`: نوع المدخلات يساعد MCP/LangChain على بناء schema للأداة.
- `mcp.run(transport="streamable-http")`: يشغّل السيرفر عبر HTTP على endpoint `/mcp`.

ملاحظة: الدالة الحالية مثال تجريبي وترجع نصًا ثابتًا. في تطبيق حقيقي يمكن استبدالها باستدعاء Weather API.

### mathservr.py

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

هذا السيرفر لا يفتح port ولا endpoint. الكلاينت يشغله كعملية فرعية ويتخاطب معه عبر standard input/output.

## شرح كود الكلاينت

ملف `client.py` هو نقطة التجميع. هو يتصل بأكثر من MCP Server في نفس الوقت:

```python
client = MultiServerMCPClient(
    {
        "Weather": {
            "transport": "streamable-http",
            "url": "http://localhost:8000/mcp",
        },
        "Math": {
            "transport": "stdio",
            "command": "python",
            "args": ["mathservr.py"],
        },
    }
)
```

بعد ذلك يجلب الأدوات:

```python
tools = await client.get_tools()
```

هذه الخطوة ترسل `tools/list` لكل سيرفر، ثم تحوّل أدوات MCP إلى أدوات LangChain يمكن تمريرها إلى Agent.

ثم ينشئ النموذج والـ Agent:

```python
model = ChatGroq(model="qwen/qwen3-32b")
agent = create_agent(model, tools)
```

وعند الاستدعاء:

```python
math_response = await agent.ainvoke(
    {
        "messages": [
            {"role": "user", "content": "What is (5 + 7)*8?"},
        ]
    }
)
print("Agent response:", math_response["messages"][-1].content)
```

الـ Agent يقرر هل يحتاج أداة أم لا. في سؤال الحساب، يستطيع استخدام أدوات `add` و `multiply`. وفي سؤال الطقس، يستطيع استخدام `get_weather`.

مهم: الرسائل الراجعة من LangChain تكون كائنات مثل `AIMessage`، لذلك نقرأ المحتوى بـ `.content` وليس `["content"]`.

## طريقة التواصل في هذا المشروع

التدفق الكامل كالتالي:

```text
User question
    |
    v
LangChain Agent in client.py
    |
    v
MultiServerMCPClient
    |
    +--> Weather server
    |       transport: streamable-http
    |       endpoint: http://localhost:8000/mcp
    |       tool: get_weather(location)
    |
    +--> Math server
            transport: stdio
            command: python mathservr.py
            tools: add(a, b), subtract(a, b), multiply(a, b)
```

خطوات التواصل:

1. تشغّل `weather.py` في Terminal مستقل لأنه HTTP server مستمر.
2. تشغّل `client.py` في Terminal آخر.
3. الكلاينت يتصل بسيرفر الطقس على `/mcp`.
4. الكلاينت يشغّل `mathservr.py` تلقائيًا كـ subprocess.
5. الكلاينت يطلب قائمة الأدوات من السيرفرين.
6. LangChain Agent يستخدم الأدوات المناسبة حسب سؤال المستخدم.
7. النتيجة ترجع إلى `client.py` وتُطبع في الطرفية.

## طريقة التشغيل

من داخل مجلد `MCP`:

```bash
uv run weather.py
```

اترك هذا السيرفر يعمل. يجب أن ترى شيئًا قريبًا من:

```text
Uvicorn running on http://127.0.0.1:8000
```

ثم افتح Terminal آخر من نفس مجلد `MCP` وشغّل:

```bash
uv run client.py
```

إذا ظهر تحذير مثل:

```text
VIRTUAL_ENV=.venv does not match the project environment path
```

فهو تحذير من `uv` عن البيئة الافتراضية، وليس خطأ MCP. يمكن استخدام:

```bash
uv run --active client.py
```

إذا كنت تريد إجبار `uv` على استخدام البيئة النشطة حاليًا.

## متطلبات البيئة

الاعتماديات الأساسية موجودة في `pyproject.toml`:

- `mcp`
- `langchain-mcp-adapters`
- `langchain-groq`
- `langgraph`

لأن `client.py` يستخدم Groq، يجب أن يحتوي ملف `.env` على مفتاح Groq:

```env
GROQ_API_KEY=your_key_here
```

## أخطاء شائعة

### Unknown transport: streamable-httpe

هذا خطأ إملائي. الصحيح:

```python
mcp.run(transport="streamable-http")
```

### GET / أو GET /weather يعطي 404

طبيعي. سيرفر MCP ليس موقع ويب ولا REST API. الأداة اسمها `get_weather` لكنها لا تصبح endpoint باسم `/weather`.

### GET /mcp يعطي 406

طبيعي عند فتحه من المتصفح. `/mcp` يحتاج MCP client يرسل headers ورسائل JSON-RPC مناسبة.

### AIMessage object is not subscriptable

هذا يحدث عند قراءة رسالة LangChain كأنها dict:

```python
response["messages"][-1]["content"]
```

الصحيح:

```python
response["messages"][-1].content
```

## ملاحظات أمنية

- عند استخدام `streamable-http` محليًا، الأفضل أن يبقى السيرفر على `127.0.0.1` أو `localhost`.
- لا تعرض MCP server على الإنترنت بدون authentication و authorization مناسبين.
- الأدوات التي تنفذ عمليات حساسة يجب أن تطلب تأكيد المستخدم أو تطبق صلاحيات واضحة.
- لا تجعل سيرفر `stdio` يطبع أي شيء عادي على `stdout` خارج رسائل MCP، لأن ذلك قد يفسد البروتوكول. استخدم `stderr` للـ logs.

## مصادر مختصرة

- [MCP Specification 2025-06-18: Base Protocol Overview](https://modelcontextprotocol.io/specification/2025-06-18/basic)
- [MCP Specification 2025-06-18: Transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [MCP Specification 2025-06-18: Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

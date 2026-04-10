from base64 import b64encode

from langchain_core.messages import HumanMessage

OCR_PROMPT = """
请识别图片中的题目文本，只输出识别后的纯文本内容，不要添加解释、前缀或多余说明。
如果图片中包含公式、选项、图注或题干，请尽量按原有结构完整保留。
""".strip()


class ImageOCRService:
    def __init__(self, llm) -> None:
        self.llm = llm

    def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        encoded = b64encode(image_bytes).decode('utf-8')
        message = HumanMessage(
            content=[
                {'type': 'text', 'text': OCR_PROMPT},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{encoded}'}},
            ]
        )
        result = self.llm.invoke([message])
        content = getattr(result, 'content', '')

        # Different providers may return either a plain string or a structured content list.
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get('text'):
                    parts.append(item['text'])
                elif isinstance(item, str):
                    parts.append(item)
            content = '\n'.join(parts)

        text = str(content).strip()
        if not text:
            raise ValueError('无法从图片中识别出题目文本')
        return text
import pytest

from app.models.schemas import RoutingPacket
from app.services.llm import ClinicLLMService, OpenAICompatibleProvider, build_llm_provider
from app.settings import Settings


class FakeProvider:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, *, text_response: str = "", json_response=None, error: Exception | None = None):
        self.text_response = text_response
        self.json_response = json_response or {}
        self.error = error
        self.text_calls = []
        self.json_calls = []

    async def chat_text(self, messages, temperature=None):
        self.text_calls.append((messages, temperature))
        if self.error:
            raise self.error
        return self.text_response

    async def chat_json(self, messages, temperature=None):
        self.json_calls.append((messages, temperature))
        if self.error:
            raise self.error
        return self.json_response


def test_settings_prioritize_generic_llm_config():
    settings = Settings(
        llm_provider="openai_compatible",
        llm_api_key="llm-key",
        llm_base_url="https://llm.example.com/v1",
        llm_model="llm-model",
        llm_timeout_seconds=12,
        llm_temperature=0.35,
        openai_api_key="openai-key",
        openai_base_url="https://openai.example.com/v1",
        openai_model="openai-model",
        openai_timeout_seconds=30,
        openai_temperature=0.1,
    )

    assert settings.resolved_llm_provider == "openai_compatible"
    assert settings.resolved_llm_api_key == "llm-key"
    assert settings.resolved_llm_base_url == "https://llm.example.com/v1"
    assert settings.resolved_llm_model == "llm-model"
    assert settings.resolved_llm_timeout_seconds == 12
    assert settings.resolved_llm_temperature == pytest.approx(0.35)


def test_settings_fallback_to_openai_aliases():
    settings = Settings(
        llm_api_key=None,
        llm_base_url=None,
        llm_model=None,
        llm_timeout_seconds=None,
        llm_temperature=None,
        openai_api_key="openai-key",
        openai_base_url="https://openai.example.com/v1",
        openai_model="openai-model",
        openai_timeout_seconds=45,
        openai_temperature=0.2,
    )

    assert settings.resolved_llm_api_key == "openai-key"
    assert settings.resolved_llm_base_url == "https://openai.example.com/v1"
    assert settings.resolved_llm_model == "openai-model"
    assert settings.resolved_llm_timeout_seconds == 45
    assert settings.resolved_llm_temperature == pytest.approx(0.2)


def test_build_llm_provider_rejects_unknown_provider():
    settings = Settings(llm_provider="anthropic", llm_api_key="test-key")

    with pytest.raises(ValueError, match="Unsupported llm provider"):
        build_llm_provider(settings)


def test_openai_compatible_provider_omits_temperature_for_gpt5_models():
    provider = OpenAICompatibleProvider(Settings(llm_model="gpt-5-mini"))

    request_kwargs = provider._chat_request_kwargs(
        [{"role": "user", "content": "hola"}],
        temperature=0.4,
    )

    assert "temperature" not in request_kwargs


@pytest.mark.asyncio
async def test_clinic_llm_service_uses_provider_contract_for_text():
    provider = FakeProvider(text_response="respuesta desde provider")
    service = ClinicLLMService(provider)

    reply = await service.build_conversation_reply("Hola", ["Prefiere horario matutino"])

    assert reply == "respuesta desde provider"
    assert len(provider.text_calls) == 1
    messages, temperature = provider.text_calls[0]
    assert temperature is None
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_clinic_llm_service_falls_back_when_provider_json_fails():
    provider = FakeProvider(error=RuntimeError("provider unavailable"))
    service = ClinicLLMService(provider)

    decision = await service.classify_state_route(
        RoutingPacket(
            user_message="Quiero una cita para manana",
            active_goal="conversation",
            stage="open",
        )
    )

    assert decision.next_node == "appointment"
    assert decision.reason == "heuristic-fallback"

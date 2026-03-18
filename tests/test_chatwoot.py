from app.models.schemas import ChatwootWebhook
from app.services.chatwoot import ChatwootClient
from app.settings import Settings


def test_chatwoot_webhook_exposes_account_id_from_payload():
    payload = ChatwootWebhook(
        account={"id": 17},
        conversation={"id": 321},
        content="Hola",
    )

    assert payload.account_id == "17"


def test_chatwoot_client_builds_account_scoped_messages_url():
    client = ChatwootClient(
        Settings(
            _env_file=None,
            chatwoot_reply_enabled=True,
            chatwoot_api_base_url="https://chatwoot.example.com",
            chatwoot_api_token="token",
            chatwoot_account_id="17",
        )
    )

    assert (
        client._build_messages_url("17", "321")
        == "https://chatwoot.example.com/api/v1/accounts/17/conversations/321/messages"
    )

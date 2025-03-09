# n8n Connection Mode for Google Sheets Integration

## Connection Type: Polling (No Tunnel Required)

The Google Sheets integration described in the guide works with n8n running locally (on localhost) without requiring tunnel mode. Here's why:

### How the Google Sheets Trigger Works

The Google Sheets Trigger node uses **polling** rather than webhooks. This means:

1. The n8n instance periodically checks Google's servers for changes
2. Google does not need to send data back to your n8n instance
3. All connections are initiated from n8n to Google, not the other way around

### OAuth Authentication

Although OAuth authentication requires a redirect URL, n8n handles this elegantly:

- The redirect URL `https://oauth.n8n.io/callback` points to n8n's proxy service
- This proxy service handles the OAuth flow and redirects back to your local instance
- Your n8n instance doesn't need to be publicly accessible for this to work

## When Do You Need Tunnel Mode?

Tunnel mode (`n8n start --tunnel`) is only required when:

1. You're using nodes that create webhooks that external services need to call
2. External services need to actively push data to your n8n instance
3. You're testing webhook-based integrations locally

Examples where tunnel mode would be needed:
- GitHub webhooks
- Stripe payment notifications
- Shopify order webhooks
- Custom API endpoints using the Webhook node

## Conclusion

For the Google Sheets integration described in this guide, you can run n8n normally without tunnel mode. The connection will work perfectly fine on localhost.

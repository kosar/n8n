# Connecting Google Sheets to n8n: A Step-by-Step Guide

This guide will walk you through the process of setting up a Google Sheets integration with n8n. You'll learn how to:

1. Connect n8n to your Google Sheets account
2. Set up a workflow that monitors a sheet for new rows
3. Process that data and write it back to another sheet with a timestamp

## Prerequisites

- n8n running locally (via the run-n8n.sh script)
  > **Note:** This Google Sheets integration works with n8n running on localhost without requiring tunnel mode. For details on why this works, see the [Connection Mode Clarification](./google-sheets-integration-clarification.md).
- A Google account with access to Google Sheets
- A web browser to access both n8n and Google Sheets

## Step 1: Prepare Your Google Sheets

First, let's set up the Google Sheets you'll be working with:

1. Go to [Google Sheets](https://sheets.google.com) and log in to your Google account
2. Create a new spreadsheet
3. Rename the first sheet to "RobitIn"
4. Create a second sheet in the same spreadsheet and name it "RobotOut"
5. In "RobitIn", add some column headers in the first row (e.g., "Name", "Email", "Message")
6. In "RobotOut", add a header for the first column called "Timestamp" and match the other headers from "RobitIn"

Make sure to note the spreadsheet's URL - you'll need to extract the Spreadsheet ID from it. The URL follows this format:
`https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit`

## Step 2: Set Up Google Sheets Authentication in n8n

n8n needs permission to access your Google account to read and write to your sheets:

1. Open your n8n interface (usually at http://localhost:5678)
2. Click on **Settings** in the left sidebar
3. Select **Credentials**
4. Click **+ Add Credential**
5. Find and select **Google Sheets OAuth2 API**
6. Enter a name for your credential (e.g., "My Google Sheets")

### Creating OAuth2 Credentials in Google Cloud Console

To connect n8n to Google Sheets, you need to create OAuth credentials:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services > Library**
4. Search for "Google Sheets API" and enable it
5. Navigate to **APIs & Services > Credentials**
6. Click **Create Credentials** and select **OAuth client ID**
7. Select **Web application** as the application type
8. Add a name (e.g., "n8n Integration")
9. Under "Authorized redirect URIs", add: `https://oauth.n8n.io/callback`
10. Click **Create** and note the generated Client ID and Client Secret

Return to n8n and enter these credentials:

1. Enter the Client ID and Client Secret you just created
2. Click **Connect** and follow the Google authentication process
3. Grant the requested permissions
4. Once connected, click **Save** to store your credentials in n8n

## Step 3: Create a Google Sheets Trigger Workflow

Now let's create a workflow that triggers when a new row is added to the "RobitIn" sheet:

1. In the n8n interface, click on **Workflows** in the left sidebar
2. Click **+ Add Workflow** to create a new workflow
3. Name your workflow (e.g., "Google Sheets Integration")

### Adding the Google Sheets Trigger Node

1. Click the **+** button in the center of the canvas
2. Search for "Google Sheets Trigger" and select it
3. Configure the node:
   - **Authentication**: Select the Google Sheets credential you created
   - **Trigger**: Select "Row Added"
   - **Sheet ID**: Paste your Spreadsheet ID from the Google Sheets URL
   - **Sheet Name**: Enter "RobitIn" (exactly as it appears in your spreadsheet)
   - **Detect New Rows By**: Choose "Modified Timestamp" for real-time detection
   - **Options**: Check "Include Column Names" to get column headers

Your trigger node is now set up to detect when a new row is added to the "RobitIn" sheet.

### Adding a Function Node to Add a Timestamp

Next, let's add a Function node to prepare the data for writing to the "RobotOut" sheet:

1. Click on the output of the Google Sheets Trigger node (the small circle on the right)
2. Select **Function** in the node selector
3. In the Function node, add this code:
```javascript
const newItem = items[0].json;
newItem.Timestamp = new Date().toISOString();
return { json: newItem };
```
4. Click **Execute Node** to test the function

### Adding the Google Sheets Node to Write Data

Finally, let's add a Google Sheets node to write the data to the "RobotOut" sheet:

1. Click on the output of the Function node
2. Select **Google Sheets** in the node selector
3. Configure the node:
   - **Authentication**: Select the Google Sheets credential you created
   - **Operation**: Select "Append"
   - **Sheet ID**: Paste your Spreadsheet ID from the Google Sheets URL
   - **Sheet Name**: Enter "RobotOut"
   - **Data**: Select "JSON" and map the fields accordingly

4. Click **Execute Node** to test the workflow

Your workflow is now set up to monitor the "RobitIn" sheet for new rows, add a timestamp, and write the data to the "RobotOut" sheet.

## Conclusion

You've successfully connected Google Sheets to n8n and created a workflow that processes new rows and writes them to another sheet with a timestamp. This integration can be extended to include more complex data processing and automation tasks.

You've successfully created an n8n workflow that:

1. Monitors a Google Sheet ("RobitIn") for new data
2. Processes that data and adds a timestamp
3. Writes the enhanced data to another sheet ("RobotOut")

This foundation can be built upon for more complex automations involving Google Sheets and other integrated services in n8n.

Remember that you can always revisit and enhance your workflow as your needs evolve. n8n's visual interface makes it easy to add new functionality or modify existing processes.


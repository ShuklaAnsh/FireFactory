import React, { useEffect, useState } from 'react';
import { Layout } from "./components/layout/layout";
import './App.scss';
import { FireButton } from "./components/fire-button/fire-button";
import { ServerStatus } from "./components/server-status/server-status";

function App() {

  const [serverStatus, setServerStatus] =
    useState({ready: false, msg: "API not connected."} as ServerStatus)

  const sendHeartbeat = async () => {
    try {
      const resp: Response = await fetch('/heartbeat');
      if (resp.status === 200) {
        setServerStatus({ready: true, msg: "Fire Generation is ready"})
      } else if (resp.status === 101) {
        setServerStatus({ready: false, msg: "Model is being trained. This can take several minutes. Please Wait."})
      } else if (resp.status === 100){
        setServerStatus({ready: false, msg: "Training failed. Please restart API."})
      } else {
        setServerStatus({ready: false, msg: "API not connected."});
      }
    } catch (error) {
      setServerStatus({ready: false, msg: "Failed to fetch data from API. Please restart server."});
      console.error(error);
    }
  }

  // Runs once if deps array is empty
  useEffect(() => {
    // Set interval to send heartbeat every 200ms
    setInterval(() => sendHeartbeat(), 200);
  }, []);

  return (
    <Layout>
      <ServerStatus serverStatus={serverStatus} />
      <h1>Fire Factory &#128293;</h1>
      <FireButton serverReady={serverStatus.ready}/>
    </Layout>
  );
}

export default App;

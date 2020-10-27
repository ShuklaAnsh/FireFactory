import React from "react";
import Button from "@material-ui/core/Button";
import './fire-button.scss';

export const FireButton = (props: {serverReady: boolean}) => {

  async function handleButtonClick(event: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    try {
      event.preventDefault();
      const resp = await fetch("/fire");
      if (resp.ok) {
        const respData = await resp.json();
        alert(respData)
      } else {
         alert("Could not generate fresh fire :( try again!");
      }
    } catch(err) {
      console.error(err);
      alert("Could not generate fresh fire :( try again!");
    }
  }

  return (
    <>
      <Button
        id="check_button"
        variant="contained"
        disabled={!props.serverReady}
        onClick={(
          event: React.MouseEvent<HTMLButtonElement, MouseEvent>
        ) => {
          handleButtonClick(event);
        }}
      >
        &#128293; &#128293; &#128293;
      </Button>
    </>
  )
}
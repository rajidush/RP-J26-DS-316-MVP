import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json(
    { 
      error: "Cloud API disabled. This system operates strictly offline using local LM Studio & FastAPI." 
    }, 
    { status: 410 }
  );
}
